import torch
from diffusers import DiffusionPipeline
import torch

seed_value=42
generator=torch.Generator().manual_seed(seed_value)

pipe_id = "stabilityai/stable-diffusion-3-medium-diffusers"
pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("~/fine_tuned_model", weight_name="pytorch_lora_weights.safetensors",adapter_name="trained")
pipe.fuse_lora(adapter_names=["trained"], lora_scale=0.9)

prompt = "PROMPT HERE"

lora_scale = 0.9
image = pipe(
    prompt=prompt,
    negative_prompt="",
    num_inference_steps=28,
    height=1024,
    width=1024,
    guidance_scale=7.0,
    generator=generator,
).images[0]
image.save("output.png")
