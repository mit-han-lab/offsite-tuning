# l25
MODEL="gpt2-xl"
num_student_layers=16
bs=10
pad=2


export WANDB_PROJECT="offsite_tuning"
export WANDB_NAME="${MODEL}_emulator_${num_student_layers}_${pad}_${pad}"
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --mixed_precision=bf16 --multi_gpu \
    offsite_tuning/run_clm.py \
    --model_name_or_path /dataset/gpt/$MODEL \
    --train_tokenized_dataset /dataset/pile/gpt_tokenized/00 \
    --val_tokenized_dataset /dataset/gpt_tokenized/wikitext-2-raw-v1 \
    --preprocessing_num_workers 88 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --num_warmup_steps 2000 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --lm_weight 1.0 \
    --kd_weight 30.0 \
    --seed 42 \
    --block_size 512 \
    --eval_steps 100 \
    --num_student_layers $num_student_layers \
    --student_l_pad $pad \
    --student_r_pad $pad \
    --output_dir emulators/${MODEL}/${num_student_layers}_${pad}_${pad} \
    --report_to wandb
