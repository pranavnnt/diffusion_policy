python train.py \
  --config-name=train_diffusion_unet_real_lowdim_workspace_head \
  name="new_head_training" \
  dataloader.num_workers=8 \
  optimizer.lr=1e-5 \
  training.num_epochs=50 \
