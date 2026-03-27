train:
  dataset_path: OCTDataset
  batch_size_per_gpu: 32 # Safe for 24GB VRAM with ViT-Small
  save_checkpoint_bnn: true # Saves intermediate checkpoints
  save_ckpt_freq: 20 # Save every 20 epochs
  epochs: 100
  OFFICIAL_EPOCH_LENGTH: 1250
  output_dir: ./output/oct_continual

student:
  arch: vit_small
  patch_size: 14
  drop_path_rate: 0.1
  drop_path_uniform: true

teacher:
  momentum_teacher: 0.996

optim:
  epochs: 100
  weight_decay: 0.04
  weight_decay_end: 0.4
  base_lr: 0.0005 # Lower learning rate for CONTINUAL pretraining
  warmup_epochs: 5 # Short warmup

crops:
  local_crops_number: 8
  global_crops_scale: [0.32, 1.0]
  local_crops_scale: [0.05, 0.32]