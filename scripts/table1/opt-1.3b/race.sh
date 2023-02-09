TASK="race"
MODEL="facebook/opt-1.3b"
num_student_layers=8
pad=2

bs=1
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --mixed_precision=bf16 --multi_gpu \
    offsite_tuning/run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name ${TASK} \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --lm_weight 1.0 \
    --kd_weight 0.0 \
    --seed 42 \
    --eval_steps $eval_steps \
    --block_size 600 \
    --train_module all \
    --save_module all \
    --no_teacher \
    --output_dir logs/table1/$MODEL/${TASK}/ft_all

bs=4
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --mixed_precision=bf16 --multi_gpu \
    offsite_tuning/run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name ${TASK} \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --lm_weight 1.0 \
    --kd_weight 0.0 \
    --seed 42 \
    --eval_steps $eval_steps \
    --block_size 600 \
    --num_student_layers $num_student_layers \
    --student_l_pad ${pad} \
    --student_r_pad ${pad} \
    --train_module adapter \
    --save_module all \
    --no_teacher \
    --restart_training \
    --load_student emulators/${MODEL}/${num_student_layers}_${pad}_${pad} \
    --output_dir logs/table1/${MODEL}/${TASK}/ft_distill_emulator/${num_student_layers}_${pad}_${pad}

bash scripts/table1/eval.sh ${TASK} ${MODEL} ${pad} ${num_student_layers}
