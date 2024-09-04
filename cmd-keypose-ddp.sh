export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1

GLOBAL_BATCH_SIZE=64
GLOBAL_NUM_WORKERS=16

GPUS_PER_NODE=4

PER_PROCESS_BATCH_SIZE=$(($GLOBAL_BATCH_SIZE / $GPUS_PER_NODE))
PER_PROCESS_NUM_WORKERS=$(($GLOBAL_NUM_WORKERS / $GPUS_PER_NODE))

torchrun --nproc_per_node=$GPUS_PER_NODE\
    train.py \
        --config-name train_keypose_transformer_workspace_ddp \
        task=keypose_sim_insertion_scripted \
        dataloader.batch_size=$PER_PROCESS_BATCH_SIZE \
        dataloader.num_workers=$PER_PROCESS_NUM_WORKERS \
        val_dataloader.batch_size=$PER_PROCESS_BATCH_SIZE \
        val_dataloader.num_workers=$PER_PROCESS_NUM_WORKERS
