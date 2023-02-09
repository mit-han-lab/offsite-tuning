# l25
MODEL="facebook/opt-1.3b"
bs=4
lr=1e-4

for pad in 2 4 6 8; do
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
        --mixed_precision=bf16 --multi_gpu \
        offsite_tuning/run_clm.py \
        --model_name_or_path $MODEL \
        --dataset_name wikitext \
        --dataset_config_name wikitext-2-raw-v1 \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --learning_rate $lr \
        --num_train_epochs 10 \
        --num_warmup_steps 100 \
        --lr_scheduler_type cosine \
        --lm_weight 1.0 \
        --kd_weight 0.0 \
        --no_save_model \
        --seed 42 \
        --block_size 512 \
        --eval_steps 10 \
        --student_l_pad $pad \
        --student_r_pad 0 \
        --train_module adapter \
        --output_dir logs/figure3/${MODEL}/wikitext-2-raw-v1/${pad}_0
done

for pad in 2 4 6 8; do
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
        --mixed_precision=bf16 --multi_gpu \
        offsite_tuning/run_clm.py \
        --model_name_or_path $MODEL \
        --dataset_name wikitext \
        --dataset_config_name wikitext-2-raw-v1 \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --learning_rate $lr \
        --num_train_epochs 10 \
        --num_warmup_steps 100 \
        --lr_scheduler_type cosine \
        --lm_weight 1.0 \
        --kd_weight 0.0 \
        --no_save_model \
        --seed 42 \
        --block_size 512 \
        --eval_steps 10 \
        --student_l_pad 0 \
        --student_r_pad $pad \
        --train_module adapter \
        --output_dir logs/figure3/${MODEL}/wikitext-2-raw-v1/0_${pad}
done

for pad in 1 2 3 4; do
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
        --mixed_precision=bf16 --multi_gpu \
        offsite_tuning/run_clm.py \
        --model_name_or_path $MODEL \
        --dataset_name wikitext \
        --dataset_config_name wikitext-2-raw-v1 \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --learning_rate $lr \
        --num_train_epochs 10 \
        --num_warmup_steps 100 \
        --lr_scheduler_type cosine \
        --lm_weight 1.0 \
        --kd_weight 0.0 \
        --no_save_model \
        --seed 42 \
        --block_size 512 \
        --eval_steps 10 \
        --student_l_pad $pad \
        --student_r_pad $pad \
        --train_module adapter \
        --output_dir logs/figure3/${MODEL}/wikitext-2-raw-v1/${pad}_${pad}
done
