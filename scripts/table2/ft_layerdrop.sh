TASK=$1               # "hellaswag"
MODEL=$2              # "facebook/opt-1.3b"
num_student_layers=$3 # 8
bs=$4                 # 8
pad=$5                # 2
eval_steps=$6         # 10
lr=$7                 # 5e-5

### emulator ft
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --mixed_precision=bf16 --multi_gpu \
    offsite_tuning/run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name ${TASK} \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --learning_rate $lr \
    --lr_scheduler_type cosine \
    --num_train_epochs 5 \
    --lm_weight 1.0 \
    --kd_weight 0.0 \
    --seed 42 \
    --eval_steps $eval_steps \
    --num_student_layers $num_student_layers \
    --student_l_pad ${pad} \
    --student_r_pad ${pad} \
    --train_module adapter \
    --save_module adapter \
    --no_teacher \
    --output_dir logs/table2/${MODEL}/${TASK}/ft_uniform_drop/${num_student_layers}_${pad}_${pad}
