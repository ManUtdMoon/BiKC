python eval_kt.py \
    -k data/outputs/2024.04.01/23.14.16_train_keypose_transformer_keypose_sim_transfer_cube_scripted/checkpoints/latest.ckpt \
    -t data/outputs/2024.05.23/15.19.46_train_trajectory_diffusion_unet_trajectory_sim_transfer_cube_scripted/checkpoints/latest.ckpt \
    -o data/eval/sim_transfer_cube_scripted/kt/ \
    -c diffusion_policy/config/task/sim_transfer_cube_scripted.yaml \
    -e 0.35 \
    -d cuda:3

python eval_kt.py \
    -k data/outputs/2024.04.01/23.14.16_train_keypose_transformer_keypose_sim_transfer_cube_scripted/checkpoints/latest.ckpt \
    -t data/outputs/2024.05.23/15.19.50_train_trajectory_diffusion_unet_trajectory_sim_transfer_cube_scripted/checkpoints/latest.ckpt \
    -o data/eval/sim_transfer_cube_scripted/kt/ \
    -c diffusion_policy/config/task/sim_transfer_cube_scripted.yaml \
    -e 0.35 \
    -d cuda:3

python eval_kt.py \
    -k data/outputs/2024.04.01/23.14.16_train_keypose_transformer_keypose_sim_transfer_cube_scripted/checkpoints/latest.ckpt \
    -t data/outputs/2024.05.23/15.20.48_train_trajectory_diffusion_unet_trajectory_sim_transfer_cube_scripted/checkpoints/latest.ckpt \
    -o data/eval/sim_transfer_cube_scripted/kt/ \
    -c diffusion_policy/config/task/sim_transfer_cube_scripted.yaml \
    -e 0.35 \
    -d cuda:3