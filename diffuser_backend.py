from __future__ import annotations
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import modal


diffsynth_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install(
        "git",
        "libsm6",
        "libxext6",
        "g++",
        "wget",
        "unzip"
    )
    .run_commands(
        "git clone https://github.com/yonagin/DiffSynth-Studio.git /root/DiffSynth-Studio",
        "cd /root/DiffSynth-Studio && pip install -e .",
    )
    .pip_install(
        "accelerate",
        "datasets",
        "transformers",
        "peft==0.17.0"
    )
    # 设置 PyTorch 使用 CUDA 12.4
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        "PATH": "/usr/local/cuda/bin:$PATH",
    })
)

app = modal.App(name="diffsynth-studio-app", image=diffsynth_image)


# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------

def b64_to_pil(data: str):
    from PIL import Image
    import io
    if data.startswith("data:image"):
        data = data.split(",", 1)[1]
    img_bytes = base64.b64decode(data)
    return Image.open(io.BytesIO(img_bytes))


def pil_to_b64(img) -> str:
    import io
    with io.BytesIO() as buf:
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()


# -----------------------------------------------------------------------------
# Modal 配置
# -----------------------------------------------------------------------------
VOLUME_DIR = "/models"
output_dir = Path(VOLUME_DIR) / "outputs"
volume = modal.Volume.from_name('forge-data', create_if_missing=True)


# -----------------------------------------------------------------------------
@app.cls(
    gpu="T4",
    volumes={VOLUME_DIR: volume},
    timeout=360,
    scaledown_window=10,
)
class Model:
    """
    一个有状态的类，用于加载和缓存AI模型，避免在每次请求时都重新加载。
    """

    @modal.enter()
    def setup(self):
        """
        此方法在容器首次启动时运行一次。用于初始化。
        """
        import torch

        print("--- Cold start: Initializing Model class ---")
        self.pipe = None
        self.current_model_name = None
        self.current_pipeline_type = None
        self.current_lora_name = None  # 新增：跟踪当前加载的LoRA

        if torch.cuda.is_available():
            self.torch_dtype = torch.bfloat16
            self.device = "cuda"
        else:
            self.torch_dtype = torch.float32
            self.device = "cpu"

    def _load_pipeline(self, model_name: str, **kwargs):
        """
        一个内部方法，用于智能加载或切换模型。
        只有在请求的模型与当前加载的模型不同时，才会执行加载操作。
        """
        import torch
        from diffsynth.pipelines import qwen_image
        # 导入官方示例中使用的 ModelConfig
        from diffsynth.pipelines.qwen_image import ModelConfig

        if model_name == self.current_model_name:
            print(f"Model '{model_name}' is already loaded.")
            return

        if self.pipe is not None:
            print(f"Switching model. Unloading '{self.current_model_name}'.")
            del self.pipe
            torch.cuda.empty_cache()
            self.pipe = None
            self.current_lora_name = None  # 切换基础模型时重置LoRA状态

        print(f"Loading new model: '{model_name}'...")

        model_path = Path(VOLUME_DIR) / model_name
        
        # 【重要修改】使用官方推荐的 from_pretrained 方法加载 QwenImagePipeline
        # 这种方法能正确处理多组件模型的加载，避免 state_dict 为 None 的问题。
        # 我们假设模型文件结构为 /models/{model_name}/{component_name}/...
        self.pipe = qwen_image.QwenImagePipeline.from_pretrained(
            torch_dtype=self.torch_dtype,
            device=self.device,
            model_configs=[
                ModelConfig(path=str(model_path / "transformer")),
                ModelConfig(path=str(model_path / "text_encoder")),
                ModelConfig(path=str(model_path / "vae")),
            ],
            tokenizer_config=ModelConfig(path=str(model_path / "tokenizer")),
        )
        
        self.current_model_name = model_name
        print("--- Model loaded successfully ---")

    @modal.method()
    async def text_to_image(
            self,
            model_name: str,
            prompt: str,
            negative_prompt: str = "",
            width: int = 1664,
            height: int = 928,
            num_inference_steps: int = 30,
            true_cfg_scale: float = 4.0,
            seed: int = -1,
            lora_name: Optional[str] = None,
            lora_scale: float = 0.8
    ):
        """文生图方法"""
        import torch

        self._load_pipeline(model_name)

        # --- LoRA处理逻辑 ---
        if lora_name and lora_name != self.current_lora_name:
            print(f"Loading LoRA weights: {lora_name}")
            # 清除之前的LoRA
            if hasattr(self.pipe, 'clear_lora'):
                self.pipe.clear_lora()
            
            lora_path = Path(VOLUME_DIR) / "outputs" / "qwen_image_lora" / lora_name
            
            # 使用Diffusionsynth的方式加载LoRA
            if hasattr(self.pipe, 'load_lora'):
                self.pipe.load_lora(self.pipe.model, str(lora_path), alpha=lora_scale)
                self.current_lora_name = lora_name
        elif not lora_name and self.current_lora_name is not None:
            print(f"Unloading LoRA weights: {self.current_lora_name}")
            if hasattr(self.pipe, 'clear_lora'):
                self.pipe.clear_lora()
            self.current_lora_name = None
        # --- LoRA处理结束 ---

        positive_magic = "Ultra HD, 4K, cinematic composition."

        # 设置生成参数，保持与原函数相同的参数名
        image = self.pipe(
            prompt=prompt + " " + positive_magic,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_steps=num_inference_steps,
            guidance_scale=true_cfg_scale,
            seed=seed if seed != -1 else None
        )
        
        torch.cuda.empty_cache()

        timestamp = int(time.time())
        save_path = output_dir / f"txt2img_{timestamp}.png"
        image.save(save_path)

        return pil_to_b64(image)

    @modal.method()
    async def image_to_image(
            self,
            model_name: str,
            input_image_b64: str,
            prompt: str,
            negative_prompt: str = "",
            strength: float = 0.8,
            num_inference_steps: int = 30,
            guidance_scale: float = 7.5,
            seed: int = -1,
            width: int = 1024,
            height: int = 1024,
            lora_name: Optional[str] = None,
            lora_scale: float = 0.8
    ):
        """图生图方法"""
        import torch
        from PIL import Image

        self._load_pipeline(model_name)

        # --- LoRA处理逻辑 ---
        if lora_name and lora_name != self.current_lora_name:
            print(f"Loading LoRA weights: {lora_name}")
            # 清除之前的LoRA
            if hasattr(self.pipe, 'clear_lora'):
                self.pipe.clear_lora()
            
            lora_path = Path(VOLUME_DIR) / "outputs" / "qwen_image_lora" / lora_name
            
            # 使用Diffusionsynth的方式加载LoRA
            if hasattr(self.pipe, 'load_lora'):
                self.pipe.load_lora(self.pipe.model, str(lora_path), alpha=lora_scale)
                self.current_lora_name = lora_name
        elif not lora_name and self.current_lora_name is not None:
            print(f"Unloading LoRA weights: {self.current_lora_name}")
            if hasattr(self.pipe, 'clear_lora'):
                self.pipe.clear_lora()
            self.current_lora_name = None
        # --- LoRA处理结束 ---

        init_image = b64_to_pil(input_image_b64).convert("RGB")
        
        # 对于图生图，我们需要将初始图像转换为潜在空间表示
        # 首先需要编码器
        if hasattr(self.pipe, 'vae_encoder'):
            # 使用VAE将初始图像编码为潜在表示
            import numpy as np
            from diffsynth.models.utils import load_state_dict
            
            # 预处理图像
            img_tensor = torch.tensor(np.array(init_image).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            img_tensor = (img_tensor * 2.0) - 1.0  # 归一化到[-1, 1]
            
            # 编码图像
            with torch.no_grad():
                latents = self.pipe.vae_encoder(img_tensor)
            
            # 添加噪声，strength控制保留原图的程度
            noise = torch.randn_like(latents)
            latents = latents * (1.0 - strength) + noise * strength
            
            # 使用diffusionsynth的方式进行图生图
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed if seed != -1 else None,
                width=width,
                height=height,
                input_latents=latents
            )
        else:
            # 如果没有内置的图生图功能，回退到文生图
            print("Warning: Image-to-image not directly supported by this pipeline, falling back to text-to-image with image prompt.")
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed if seed != -1 else None,
                width=width,
                height=height
            )
        
        torch.cuda.empty_cache()

        timestamp = int(time.time())
        save_path = output_dir / f"img2img_{timestamp}.png"
        image.save(save_path)

        return pil_to_b64(image)


# -----------------------------------------------------------------------------
# 获取可用模型列表的函数
# -----------------------------------------------------------------------------
@app.function(volumes={VOLUME_DIR: volume})
async def get_available_models():
    """获取volume中的可用模型列表"""
    models_path = Path(VOLUME_DIR)
    models = []

    if models_path.exists():
        for item in models_path.iterdir():
            if item.is_dir() and item.name != "outputs":  # 排除outputs目录
                has_model = any(
                    file.suffix in ['.safetensors', '.bin', '.ckpt', '.pth']
                    for file in item.rglob('*')
                )
                if has_model or (item / 'model_index.json').exists():
                    models.append(item.name)

    return sorted(models)


@app.function(volumes={VOLUME_DIR: volume})
async def get_available_loras():
    """获取volume中指定目录的可用LoRA模型列表"""
    loras_path = Path(VOLUME_DIR) / "outputs" / "qwen_image_lora"
    loras = []

    if loras_path.exists() and loras_path.is_dir():
        for item in loras_path.iterdir():
            if item.is_file() and item.suffix in ['.safetensors', '.bin', '.pt', '.pth']:
                loras.append(item.name)

    return sorted(loras)