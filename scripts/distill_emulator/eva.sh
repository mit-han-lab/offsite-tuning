MODEL="eva_clip_vis_enc_sz224"
bs=32
num_student_layers=22

pad=4

export WANDB_PROJECT="offsite_tuning"
export WANDB_NAME="${MODEL}_emulator_1k_${num_student_layers}_${pad}_${pad}"
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --mixed_precision=bf16 --multi_gpu \
    offsite_tuning/run_image_classification.py \
    --use_pt_imagefolder \
    --num_workers 10 \
    --model_name_or_path models/$MODEL \
    --train_dir /dataset/imagenet/train \
    --validation_dir /dataset/imagenet/val \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --num_warmup_steps 80000 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --seed 42 \
    --ignore_mismatched_sizes \
    --lm_weight 0.0 \
    --kd_weight 1.0 \
    --seed 42 \
    --eval_steps 200 \
    --num_student_layers $num_student_layers \
    --student_l_pad $pad \
    --student_r_pad $pad \
    --train_module student \
    --select_by_kd \
    --output_dir emulators/${MODEL}/${num_student_layers}_${pad}_${pad} \
    --report_to wandb