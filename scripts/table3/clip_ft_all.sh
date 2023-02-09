MODEL="CLIP-ViT-H-14-laion2B-s32B-b79K"
bs=32


DATASETS="stanford_car flowers102 pets food101"
for DATASET in $DATASETS; do
    for lr in 3e-5 2e-5 1e-5; do
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
            --mixed_precision=bf16 --multi_gpu \
            offsite_tuning/run_image_classification.py \
            --model_name_or_path laion/$MODEL \
            --train_dir dataset/$DATASET/train \
            --validation_dir dataset/$DATASET/val \
            --per_device_train_batch_size $bs \
            --per_device_eval_batch_size $bs \
            --learning_rate $lr \
            --lr_scheduler_type cosine \
            --num_warmup_steps 1000 \
            --num_train_epochs 20 \
            --no_save_model \
            --seed 42 \
            --ignore_mismatched_sizes \
            --lm_weight 1.0 \
            --kd_weight 0.0 \
            --seed 42 \
            --eval_steps 100 \
            --train_module all \
            --classifier_lr_multiplier 10.0 \
            --output_dir logs/table3/${MODEL}/ft_all/${DATASET}_${lr}
    done
done

DATASETS="cifar10 cifar100"
for DATASET in $DATASETS; do
    for lr in 3e-5 2e-5 1e-5; do
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
            --mixed_precision=bf16 --multi_gpu \
            offsite_tuning/run_image_classification.py \
            --model_name_or_path laion/$MODEL \
            --dataset_name $DATASET \
            --per_device_train_batch_size $bs \
            --per_device_eval_batch_size $bs \
            --learning_rate $lr \
            --lr_scheduler_type cosine \
            --num_warmup_steps 1000 \
            --num_train_epochs 20 \
            --no_save_model \
            --seed 42 \
            --ignore_mismatched_sizes \
            --lm_weight 1.0 \
            --kd_weight 0.0 \
            --seed 42 \
            --eval_steps 100 \
            --train_module all \
            --classifier_lr_multiplier 10.0 \
            --output_dir logs/table3/${MODEL}/ft_all/${DATASET}_${lr}
    done
done