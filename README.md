# Quantized-Training-of-SD3

**Introduction**

Stable Diffusion 3 (SD3) Medium is the most advanced text-to-image model that stability.ai has released. It’s smaller than other models, such as SDXL, yet still produces high-quality images, understands complex prompts and performs inference quickly. Despite its smaller size, fine-tuning SD3 Medium out of the box on a GPU with 16GB VRAM isn’t possible. GPUs with more than 16GB VRAM cost significantly more, whether you’re buying a GPU directly or using it through a cloud service.

Fortunately, quantizing one of the text encoders can significantly reduce the memory used during fine-tuning, allowing for customisation on a 16GB VRAM GPU. This drastically reduces costs and increases the accessibility of model customisation. We also used LoRA (Low-Rank Adaptation of Large Language Models) to further reduce VRAM usage during fine-tuning.

This repo provides you with all the files and steps needed to achieve this. For reference, I fine-tuned my model on a gd4n.2xlarge instance on AWS, which has one GPU (16GB VRAM) and 16 vCPUs (32GB RAM).

**Install GCC and G++ 9.5.0**

```
sudo apt-get install gcc-9 g++-9
```

**Install Conda**

The first step is to ensure that you have Conda installed; I used the lightweight installer [Miniforge](https://github.com/conda-forge/miniforge).

**Required Files**

Clone the GitHub repository with the required files.
```
git clone https://github.com/FilippoO2/Quantized-Training-of-SD3.git
cd Quantized-Training-of-SD3
```
**Create a Conda Environment**

Create a Conda environment (train_SD3) from the conda_config.yaml file.

```
conda env create -f conda_config.yaml
conda activate train_SD3
```

`diffusers` and `diffusers-0.30.0.dev0.dist-info` contain changes that are required for the quantization to work. Place these into your conda environment's site-packages (e.g. ~/miniforge/envs/train_SD3/lib/Python-3.12/site-packages/). You can do this using:

```
mv diffusers/ ~/miniforge3/envs/train_SD3/lib/python3.12/site-packages/
mv diffusers-0.30.0.dev0.dist-info/ ~/miniforge3/envs/train_SD3/lib/python3.12/site-packages/
```

Note: If you have not used miniforge, the destination path will be slightly different.

**Accessing SD3 Medium from Hugging Face**

Head over to SD3 Medium on Hugging Face where you must create an account if you don't already have one and agree to the SD3 Medium license.

Next, go to your profile settings in Hugging Face and select Access Tokens from the left menu. Create and copy a token, which you can then use to log in to Hugging Face through your terminal with the huggingface-cli login command. This should now be stored in ~/.cache/huggingface/token

**Configure Accelerate**

Next, we must configure accelerate. This is done by running accelerate config but editing the file directly at ~/.cache/huggingface/accelerate/default_config.yaml is easier. Notet that you might have to run and complete accelerate config for accelerate to appear in the cache.

```
mv default_config.yaml ~/.cache/huggingface/accelerate/
```

We will train our model using the train_dreambooth_lora_sd3.pyscript that has been adapted to reduce memory usage. The largest text encoder, text_encoder_3, and its tokenizer have been quantized. Text encoder 1 and 2 remain unchanged.

**Fine-tuning the Model**

Specify the model being used (MODEL_NAME), where the training images are located (INSTANCE_DIR), and where to save our model (OUTPUT_DIR).

```
export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export INSTANCE_DIR="path/to/training_images"
export OUTPUT_DIR="./fine_tuned_model"
```

We can now begin training! Make sure to change instance_prompt to the appropriate prompt for your images.

```
accelerate launch train_dreambooth_lora_sd3.py \
--pretrained_model_name_or_path=${MODEL_NAME} \
--instance_data_dir=${INSTANCE_DIR} \
--output_dir=${OUTPUT_DIR} \
--mixed_precision="bf16" \
--instance_prompt "PROMPT FOR TRAINING IMAGES" \
--resolution=512 \
--train_batch_size=4 \
--gradient_accumulation_steps=4 \
--learning_rate=0.001 \
--report_to="wandb" \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--max_train_steps=1000 \
--weighting_scheme="logit_normal" \
--seed="42" \
--use_8bit_adam \
--gradient_checkpointing \
--prior_generation_precision="bf16"
```

**Running Inference**

It may take some time, but your model should train and output a .safetensors file in your OUTPUT_DIR. Before we can test it, we need to add config.json to the OUTPUT_DIR:

```
mv config.json ${OUTPUT_DIR}
```

You can now run inference with your new model (run_trained.py):

```
python run_trained.py
```

You can adjust the balance between the original and fine-tuned model by changing lora_scale. Increasing the value of the scale produces results more similar to the fine-tuned examples, whereas a lower scale value returns an image more similar to the base SD3 Medium output.


Adapted from [diffusers](https://github.com/huggingface/diffusers).

Full guide available [here](https://medium.com/@filipposantiano/fine-tuning-stable-diffusion-3-medium-with-16gb-vram-36f4e0d084e7).
