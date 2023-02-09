TASK="sciq"
MODEL="bigscience/bloom-7b1"
num_student_layers=18
bs=1
pad=2
eval_steps=500

### emulator ft
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --mixed_precision=bf16 --multi_gpu \
    offsite_tuning/run_clm.py \
    --model_name_or_path /dataset/$FAMILY/$MODEL \
    --dataset_name ${TASK} \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --num_train_epochs 5 \
    --lm_weight 1.0 \
    --kd_weight 0.0 \
    --seed 42 \
    --block_size 600 \
    --eval_steps $eval_steps \
    --num_student_layers $num_student_layers \
    --student_l_pad ${pad} \
    --student_r_pad ${pad} \
    --train_module adapter \
    --save_module adapter \
    --no_teacher \
    --output_dir logs/${MODEL}/${TASK}/ft_uniform_drop/${num_student_layers}_${pad}_${pad}

bash scripts/table2/eval_layerdrop.sh ${TASK} ${MODEL} ${pad} ${num_student_layers}

