import torch
from diffusers import DiffusionPipeline
import torch

seed_value=69
generator=torch.Generator().manual_seed(seed_value)

pipe_id = "stabilityai/stable-diffusion-3-medium-diffusers"
pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("/home/ubuntu/trained_model_delete/", weight_name="model.safetensors",adapter_name="trained")
pipe.fuse_lora(adapter_names=["trained"], lora_scale=0.9)

prompt = "Back, left, right and front views of a toy droid"

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
image.save("lora_newest.png")