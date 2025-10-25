setx CUDA_VISIBLE_DEVICES "1"

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ^
  sdxl_train_network.py ^
  --pretrained_model_name_or_path="../models/sd_xl_base_1.0.safetensors" ^
  --train_data_dir="../dataset/train/mainchar" ^
  --output_dir="../output_models" ^
  --logging_dir="../logs" ^
  --output_name="karina" ^
  --network_module=networks.lora ^
  --network_dim=32 ^
  --network_alpha=16 ^
  --learning_rate=1e-4 ^
  --optimizer_type="AdamW8bit" ^
  --lr_scheduler="cosine" ^
  --lr_warmup_steps=100 ^
  --max_train_epochs=15 ^
  --save_every_n_epochs=1 ^
  --mixed_precision="bf16" ^
  --save_precision="bf16" ^
  --cache_latents ^
  --cache_latents_to_disk ^
  --gradient_checkpointing ^
  --xformers ^
  --seed=47 ^
  --bucket_no_upscale ^
  --min_bucket_reso=512 ^
  --max_bucket_reso=2048 ^
  --bucket_reso_steps=64 ^
  --resolution="1024,1024" ^
  --network_train_unet_only ^
  --cache_text_encoder_outputs