MODEL="gpt2-xl"
num_student_layers=16
pad=2

MODEL="facebook/opt-1.3b"
num_student_layers=8
pad=2

CUDA_VISIBLE_DEVICES="0" accelerate launch \
    --mixed_precision=bf16 --multi_gpu \
    offsite_tuning/run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --weight_decay 0.1 \
    --lm_weight 1.0 \
    --kd_weight 0.0 \
    --no_save_model \
    --seed 42 \
    --block_size 512 \
    --eval_steps 1000000 \
    --train_module all \
    --no_teacher \
    --output_dir logs/table6/${MODEL}/ft_all

CUDA_VISIBLE_DEVICES="0" accelerate launch \
    --mixed_precision=bf16 --multi_gpu \
    offsite_tuning/run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --lm_weight 1.0 \
    --kd_weight 0.0 \
    --no_save_model \
    --seed 42 \
    --block_size 512 \
    --eval_steps 100000 \
    --num_student_layers $num_student_layers \
    --student_l_pad 2 \
    --student_r_pad 2 \
    --train_module adapter \
    --no_teacher \
    --restart_training \
    --load_student emulators/${MODEL}/${num_student_layers}_${pad}_${pad} \
    --output_dir logs/table6/${MODEL}/${num_student_layers}_${pad}_${pad}_adapter


CUDA_VISIBLE_DEVICES="0" accelerate launch \
    --mixed_precision=bf16 --multi_gpu \
    offsite_tuning/run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --weight_decay 0.1 \
    --lm_weight 1.0 \
    --kd_weight 0.0 \
    --no_save_model \
    --seed 42 \
    --block_size 512 \
    --eval_steps 1000000 \
    --train_module all \
    --use_lora \
    --no_teacher \
    --output_dir logs/table6/${MODEL}/lora


CUDA_VISIBLE_DEVICES="0" accelerate launch \
    --mixed_precision=bf16 --multi_gpu \
    offsite_tuning/run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --lm_weight 1.0 \
    --kd_weight 0.0 \
    --no_save_model \
    --seed 42 \
    --block_size 512 \
    --eval_steps 100000 \
    --num_student_layers $num_student_layers \
    --student_l_pad 2 \
    --student_r_pad 2 \
    --train_module adapter \
    --no_teacher \
    --restart_training \
    --use_lora \
    --load_student emulators/${MODEL}/${num_student_layers}_${pad}_${pad} \
    --output_dir logs/table6/${MODEL}/${num_student_layers}_${pad}_${pad}_adapter_lora
