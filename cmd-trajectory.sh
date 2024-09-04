# first change policy.obs_encoder.crop_shape to [110,150] in the config file
# then run the following commands
# remember to check policy.ema_scale.total_training_steps for CM

# diffusion
## screwdriver
OMP_NUM_THREADS=1 python train.py \
    --config-name train_trajectory_diffusion_unet_workspace \
    task=trajectory_aloha_screwdriver \
    policy.obs_encoder.imagenet_norm=True \
    training.lr_warmup_steps=1000 \
    training.num_epochs=1005 \
    training.device=cuda:0 \
    training.seed=42 \
    task.dataset.seed=42 \
    logging.project=trajectory_aloha_real

## starbucks
OMP_NUM_THREADS=1 python train.py \
    --config-name train_trajectory_diffusion_unet_workspace \
    task=trajectory_aloha_starbucks \
    policy.obs_encoder.imagenet_norm=True \
    training.lr_warmup_steps=1000 \
    training.num_epochs=1005 \
    training.device=cuda:0 \
    training.seed=42 \
    task.dataset.seed=42 \
    logging.project=trajectory_aloha_real

# consistency
## screwdriver
OMP_NUM_THREADS=1 python train.py \
    --config-name train_trajectory_consistency_unet_workspace \
    task=trajectory_aloha_screwdriver \
    policy.ema_scale.total_training_steps=529000 \
    policy.obs_encoder.imagenet_norm=True \
    training.lr_warmup_steps=1000 \
    training.num_epochs=1005 \
    training.device=cuda:0 \
    training.seed=42 \
    task.dataset.seed=42 \
    logging.project=trajectory_aloha_real

## starbucks
OMP_NUM_THREADS=1 python train.py \
    --config-name train_trajectory_consistency_unet_workspace \
    task=trajectory_aloha_starbucks \
    policy.ema_scale.total_training_steps=336000 \
    policy.obs_encoder.imagenet_norm=True \
    training.lr_warmup_steps=1000 \
    training.num_epochs=1005 \
    training.device=cuda:0 \
    training.seed=42 \
    task.dataset.seed=42 \
    logging.project=trajectory_aloha_real

# keypose
## screwdriver
OMP_NUM_THREADS=1 python train.py \
    --config-name train_keypose_transformer_workspace \
    task=keypose_aloha_screwdriver

## starbucks
OMP_NUM_THREADS=1 python train.py \
    --config-name train_keypose_transformer_workspace \
    task=keypose_aloha_starbucks