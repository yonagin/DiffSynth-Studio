import torch
from PIL import Image
from typing import Union
from PIL import Image
from tqdm import tqdm
from einops import rearrange
import numpy as np

from ..models import ModelManager, load_state_dict
from ..models.qwen_image_dit import QwenImageDiT
from ..models.qwen_image_text_encoder import QwenImageTextEncoder
from ..models.qwen_image_vae import QwenImageVAE
from ..models.qwen_image_controlnet import QwenImageBlockWiseControlNet
from ..schedulers import FlowMatchScheduler
from ..utils import BasePipeline, ModelConfig, PipelineUnitRunner, PipelineUnit
from ..lora import GeneralLoRALoader
from .flux_image_new import ControlNetInput

from ..vram_management import gradient_checkpoint_forward, enable_vram_management, AutoWrappedModule, AutoWrappedLinear


class QwenImageBlockwiseMultiControlNet(torch.nn.Module):
    def __init__(self, models: list[QwenImageBlockWiseControlNet]):
        super().__init__()
        if not isinstance(models, list):
            models = [models]
        self.models = torch.nn.ModuleList(models)

    def preprocess(self, controlnet_inputs: list[ControlNetInput], conditionings: list[torch.Tensor], **kwargs):
        processed_conditionings = []
        for controlnet_input, conditioning in zip(controlnet_inputs, conditionings):
            conditioning = rearrange(conditioning, "B C (H P) (W Q) -> B (H W) (C P Q)", P=2, Q=2)
            model_output = self.models[controlnet_input.controlnet_id].process_controlnet_conditioning(conditioning)
            processed_conditionings.append(model_output)
        return processed_conditionings

    def blockwise_forward(self, image, conditionings: list[torch.Tensor], controlnet_inputs: list[ControlNetInput], progress_id, num_inference_steps, block_id, **kwargs):
        res = 0
        for controlnet_input, conditioning in zip(controlnet_inputs, conditionings):
            progress = (num_inference_steps - 1 - progress_id) / max(num_inference_steps - 1, 1)
            if progress > controlnet_input.start + (1e-4) or progress < controlnet_input.end - (1e-4):
                continue
            model_output = self.models[controlnet_input.controlnet_id].blockwise_forward(image, conditioning, block_id)
            res = res + model_output * controlnet_input.scale
        return res


class QwenImagePipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        from transformers import Qwen2Tokenizer, Qwen2VLProcessor
        
        self.scheduler = FlowMatchScheduler(sigma_min=0, sigma_max=1, extra_one_step=True, exponential_shift=True, exponential_shift_mu=0.8, shift_terminal=0.02)
        self.text_encoder: QwenImageTextEncoder = None
        self.dit: QwenImageDiT = None
        self.vae: QwenImageVAE = None
        self.blockwise_controlnet: QwenImageBlockwiseMultiControlNet = None
        self.tokenizer: Qwen2Tokenizer = None
        self.processor: Qwen2VLProcessor = None
        self.unit_runner = PipelineUnitRunner()
        self.in_iteration_models = ("dit", "blockwise_controlnet")
        self.units = [
            QwenImageUnit_ShapeChecker(),
            QwenImageUnit_NoiseInitializer(),
            QwenImageUnit_InputImageEmbedder(),
            QwenImageUnit_Inpaint(),
            QwenImageUnit_PromptEmbedder(),
            QwenImageUnit_EntityControl(),
            QwenImageUnit_BlockwiseControlNet(),
            QwenImageUnit_EditImageEmbedder(),
        ]
        self.model_fn = model_fn_qwen_image
        
        
    def load_lora(self, module, path, alpha=1):
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        loader.load(module, lora, alpha=alpha)


    def clear_lora(self):
        for name, module in self.named_modules():
            if isinstance(module, AutoWrappedLinear): 
                if hasattr(module, "lora_A_weights"):
                    module.lora_A_weights.clear()
                if hasattr(module, "lora_B_weights"):
                    module.lora_B_weights.clear()


    def training_loss(self, **inputs):
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        
        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        
        noise_pred = self.model_fn(**inputs, timestep=timestep)
        
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)
        return loss
    
    
    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5, enable_dit_fp8_computation=False):
        self.vram_management_enabled = True
        if vram_limit is None:
            vram_limit = self.get_vram()
        vram_limit = vram_limit - vram_buffer
        
        if self.text_encoder is not None:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRotaryEmbedding, Qwen2RMSNorm
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    Qwen2_5_VLRotaryEmbedding: AutoWrappedModule,
                    Qwen2RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            from ..models.qwen_image_dit import RMSNorm
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            if not enable_dit_fp8_computation:
                enable_vram_management(
                    self.dit,
                    module_map = {
                        RMSNorm: AutoWrappedModule,
                        torch.nn.Linear: AutoWrappedLinear,
                    },
                    module_config = dict(
                        offload_dtype=dtype,
                        offload_device="cpu",
                        onload_dtype=dtype,
                        onload_device=device,
                        computation_dtype=self.torch_dtype,
                        computation_device=self.device,
                    ),
                    vram_limit=vram_limit,
                )
            else:
                enable_vram_management(
                    self.dit,
                    module_map = {
                        RMSNorm: AutoWrappedModule,
                    },
                    module_config = dict(
                        offload_dtype=dtype,
                        offload_device="cpu",
                        onload_dtype=dtype,
                        onload_device=device,
                        computation_dtype=self.torch_dtype,
                        computation_device=self.device,
                    ),
                    vram_limit=vram_limit,
                )
                enable_vram_management(
                    self.dit,
                    module_map = {
                        torch.nn.Linear: AutoWrappedLinear,
                    },
                    module_config = dict(
                        offload_dtype=dtype,
                        offload_device="cpu",
                        onload_dtype=dtype,
                        onload_device=device,
                        computation_dtype=dtype,
                        computation_device=self.device,
                    ),
                    vram_limit=vram_limit,
                )
        if self.vae is not None:
            from ..models.qwen_image_vae import QwenImageRMS_norm
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    QwenImageRMS_norm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.blockwise_controlnet is not None:
            enable_vram_management(
                self.blockwise_controlnet,
                module_map = {
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
    
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
        processor_config: ModelConfig = None,
    ):
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary()
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Initialize pipeline
        pipe = QwenImagePipeline(device=device, torch_dtype=torch_dtype)
        pipe.text_encoder = model_manager.fetch_model("qwen_image_text_encoder")
        pipe.dit = model_manager.fetch_model("qwen_image_dit")
        pipe.vae = model_manager.fetch_model("qwen_image_vae")
        pipe.blockwise_controlnet = QwenImageBlockwiseMultiControlNet(model_manager.fetch_model("qwen_image_blockwise_controlnet", index="all"))
        if tokenizer_config is not None and pipe.text_encoder is not None:
            tokenizer_config.download_if_necessary()
            from transformers import Qwen2Tokenizer
            pipe.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_config.path)
        if processor_config is not None:
            processor_config.download_if_necessary()
            from transformers import Qwen2VLProcessor
            pipe.processor = Qwen2VLProcessor.from_pretrained(processor_config.path)
        return pipe
    
    
    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 4.0,
        # Image
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        # Inpaint
        inpaint_mask: Image.Image = None,
        inpaint_blur_size: int = None,
        inpaint_blur_sigma: float = None,
        # Shape
        height: int = 1328,
        width: int = 1328,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 30,
        # Blockwise ControlNet
        blockwise_controlnet_inputs: list[ControlNetInput] = None,
        # EliGen
        eligen_entity_prompts: list[str] = None,
        eligen_entity_masks: list[Image.Image] = None,
        eligen_enable_on_negative: bool = False,
        # Edit Image
        edit_image: Image.Image = None,
        # FP8
        enable_fp8_attention: bool = False,
        # Tile
        tiled: bool = False,
        tile_size: int = 128,
        tile_stride: int = 64,
        # Progress bar
        progress_bar_cmd = tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, dynamic_shift_len=(height // 16) * (width // 16))
        
        # Parameters
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "inpaint_mask": inpaint_mask, "inpaint_blur_size": inpaint_blur_size, "inpaint_blur_sigma": inpaint_blur_sigma,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "enable_fp8_attention": enable_fp8_attention,
            "num_inference_steps": num_inference_steps,
            "blockwise_controlnet_inputs": blockwise_controlnet_inputs,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "eligen_entity_prompts": eligen_entity_prompts, "eligen_entity_masks": eligen_entity_masks, "eligen_enable_on_negative": eligen_enable_on_negative,
            "edit_image": edit_image,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep, progress_id=progress_id)
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep, progress_id=progress_id)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
        
        # Decode
        self.load_models_to_device(['vae'])
        image = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        return image



class QwenImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width"))

    def process(self, pipe: QwenImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}



class QwenImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "seed", "rand_device"))

    def process(self, pipe: QwenImagePipeline, height, width, seed, rand_device):
        noise = pipe.generate_noise((1, 16, height//8, width//8), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return {"noise": noise}



class QwenImageUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: QwenImagePipeline, input_image, noise, tiled, tile_size, tile_stride):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(['vae'])
        image = pipe.preprocess_image(input_image).to(device=pipe.device, dtype=pipe.torch_dtype)
        input_latents = pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": input_latents}



class QwenImageUnit_Inpaint(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("inpaint_mask", "height", "width", "inpaint_blur_size", "inpaint_blur_sigma"),
        )

    def process(self, pipe: QwenImagePipeline, inpaint_mask, height, width, inpaint_blur_size, inpaint_blur_sigma):
        if inpaint_mask is None:
            return {}
        inpaint_mask = pipe.preprocess_image(inpaint_mask.convert("RGB").resize((width // 8, height // 8)), min_value=0, max_value=1)
        inpaint_mask = inpaint_mask.mean(dim=1, keepdim=True)
        if inpaint_blur_size is not None and inpaint_blur_sigma is not None:
            from torchvision.transforms import GaussianBlur
            blur = GaussianBlur(kernel_size=inpaint_blur_size * 2 + 1, sigma=inpaint_blur_sigma)
            inpaint_mask = blur(inpaint_mask)
        return {"inpaint_mask": inpaint_mask}


class QwenImageUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            input_params=("edit_image",),
            onload_model_names=("text_encoder",)
        )
        
    def extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def process(self, pipe: QwenImagePipeline, prompt, edit_image=None) -> dict:
        if pipe.text_encoder is not None:
            prompt = [prompt]
            # If edit_image is None, use the default template for Qwen-Image, otherwise use the template for Qwen-Image-Edit
            if edit_image is None:
                template = "
