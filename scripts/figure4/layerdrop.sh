MODEL="facebook/opt-1.3b"
num_student_layers_list="20 18 16 14 12 10 8 6 4 2"
bs=3

pad=2
for num_student_layers in $num_student_layers_list; do
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
        --mixed_precision=bf16 --multi_gpu \
        offsite_tuning/run_clm.py \
        --model_name_or_path $MODEL \
        --dataset_name wikitext \
        --dataset_config_name wikitext-2-raw-v1 \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --learning_rate 1e-4 \
        --num_train_epochs 1 \
        --num_warmup_steps 100 \
        --lr_scheduler_type cosine \
        --lm_weight 1.0 \
        --kd_weight 0.0 \
        --no_save_model \
        --seed 42 \
        --block_size 512 \
        --eval_steps 20 \
        --num_student_layers $num_student_layers \
        --student_l_pad ${pad} \
        --student_r_pad ${pad} \
        --train_module adapter \
        --output_dir logs/figure4/layerdrop/${MODEL}/wikitext-2-raw-v1/${num_student_layers}_${pad}_${pad}
done
