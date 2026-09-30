from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class QwenVLBackend:
    """Thin local inference wrapper around Qwen3-VL."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        max_pixels: int = 786_432,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        torch_dtype = getattr(torch, dtype)
        if device == "cpu" and torch_dtype == torch.float16:
            torch_dtype = torch.float32

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id)
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.max_pixels = max_pixels
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, image_path: Path, prompt: str, max_new_tokens: int | None = None) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        trimmed = generated[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

