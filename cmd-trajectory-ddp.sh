export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=2,3

GLOBAL_BATCH_SIZE=64
GLOBAL_NUM_WORKERS=16

GPUS_PER_NODE=2

PER_PROCESS_BATCH_SIZE=$(($GLOBAL_BATCH_SIZE / $GPUS_PER_NODE))
PER_PROCESS_NUM_WORKERS=$(($GLOBAL_NUM_WORKERS / $GPUS_PER_NODE))

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=25641 \
    train.py \
        --config-name train_trajectory_consistency_unet_workspace_ddp \
        task=trajectory_aloha_screwdriver \
        dataloader.batch_size=$PER_PROCESS_BATCH_SIZE \
        dataloader.num_workers=$PER_PROCESS_NUM_WORKERS \
        val_dataloader.batch_size=$PER_PROCESS_BATCH_SIZE \
        val_dataloader.num_workers=$PER_PROCESS_NUM_WORKERS
