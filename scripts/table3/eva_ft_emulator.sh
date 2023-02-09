MODEL="eva_clip_vis_enc_sz224"
bs=16
num_student_layers=20
pad=4
lr_list="1e-4 5e-5 2e-4"

for lr in $lr_list; do
    DATASETS="stanford_car aircraft flowers102 cub200 pets food101"
    for DATASET in $DATASETS; do
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
            --mixed_precision=bf16 --multi_gpu \
            offsite_tuning/run_image_classification.py \
            --model_name_or_path models/$MODEL \
            --train_dir dataset/$DATASET/train \
            --use_pt_imagefolder \
            --num_workers 10 \
            --validation_dir dataset/$DATASET/val \
            --per_device_train_batch_size $bs \
            --per_device_eval_batch_size $bs \
            --learning_rate $lr \
            --lr_scheduler_type cosine \
            --num_train_epochs 50 \
            --no_save_model \
            --seed 42 \
            --ignore_mismatched_sizes \
            --lm_weight 1.0 \
            --kd_weight 0.0 \
            --seed 42 \
            --eval_steps 200 \
            --weight_decay 0.1 \
            --num_student_layers $num_student_layers \
            --student_l_pad $pad \
            --student_r_pad $pad \
            --train_module adapter \
            --classifier_lr_multiplier 10.0 \
            --restart_training \
            --load_student emulators/${MODEL}/${num_student_layers}_${pad}_${pad} \
            --output_dir logs/table3/${MODEL}/ft_emulator/${num_student_layers}_${pad}_${pad}/${DATASET}_lr=${lr}
    done

    DATASETS="cifar10 cifar100"
    for DATASET in $DATASETS; do
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
            --mixed_precision=bf16 --multi_gpu \
            offsite_tuning/run_image_classification.py \
            --model_name_or_path models/$MODEL \
            --dataset_name $DATASET \
            --num_workers 10 \
            --per_device_train_batch_size $bs \
            --per_device_eval_batch_size $bs \
            --learning_rate $lr \
            --lr_scheduler_type cosine \
            --num_train_epochs 50 \
            --no_save_model \
            --seed 42 \
            --ignore_mismatched_sizes \
            --lm_weight 1.0 \
            --kd_weight 0.0 \
            --seed 42 \
            --eval_steps 100 \
            --weight_decay 0.1 \
            --num_student_layers $num_student_layers \
            --student_l_pad $pad \
            --student_r_pad $pad \
            --train_module adapter \
            --classifier_lr_multiplier 10.0 \
            --restart_training \
            --load_student emulators/${MODEL}/${num_student_layers}_${pad}_${pad} \
            --output_dir logs/table3/${MODEL}/ft_emulator/${num_student_layers}_${pad}_${pad}/${DATASET}_lr=${lr}
    done
done
