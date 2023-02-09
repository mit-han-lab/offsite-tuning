MODEL="facebook/opt-1.3b"
num_student_layers=8
bs=8
pad=2
lr=1e-4

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --mixed_precision=bf16 --multi_gpu \
    offsite_tuning/run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --learning_rate $lr \
    --num_train_epochs 50 \
    --num_warmup_steps 100 \
    --lr_scheduler_type cosine \
    --lm_weight 1.0 \
    --kd_weight 0.0 \
    --no_save_model \
    --seed 42 \
    --block_size 512 \
    --eval_steps 100 \
    --num_student_layers $num_student_layers \
    --student_l_pad 2 \
    --student_r_pad 2 \
    --train_module adapter \
    --restart_training \
    --use_bitfit \
    --load_student emulators/${MODEL}/${num_student_layers}_${pad}_${pad} \
    --output_dir logs/table5/${MODEL}/${num_student_layers}_2_2_bitfit/wikitext-2-raw-v1



CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --mixed_precision=bf16 --multi_gpu \
    offsite_tuning/run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --learning_rate $lr \
    --num_train_epochs 50 \
    --num_warmup_steps 100 \
    --lr_scheduler_type cosine \
    --lm_weight 1.0 \
    --kd_weight 0.0 \
    --no_save_model \
    --seed 42 \
    --block_size 512 \
    --eval_steps 100 \
    --num_student_layers $num_student_layers \
    --student_l_pad 2 \
    --student_r_pad 2 \
    --train_module adapter \
    --restart_training \
    --use_lora \
    --load_student emulators/${MODEL}/${num_student_layers}_${pad}_${pad} \
    --output_dir logs/table5/${MODEL}/${num_student_layers}_2_2_lora/wikitext-2-raw-v1

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --mixed_precision=bf16 --multi_gpu \
    offsite_tuning/run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --learning_rate $lr \
    --num_train_epochs 50 \
    --num_warmup_steps 100 \
    --lr_scheduler_type cosine \
    --lm_weight 1.0 \
    --kd_weight 0.0 \
    --no_save_model \
    --seed 42 \
    --block_size 512 \
    --eval_steps 100 \
    --num_student_layers $num_student_layers \
    --student_l_pad 2 \
    --student_r_pad 2 \
    --train_module adapter \
    --restart_training \
    --use_adapter \
    --load_student emulators/${MODEL}/${num_student_layers}_${pad}_${pad} \
    --output_dir logs/table5/${MODEL}/${num_student_layers}_2_2_adapter/wikitext-2-raw-v1

