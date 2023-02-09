TASK=$1               # "hellaswag"
MODEL=$2              # "facebook/opt-1.3b"
pad=$3                # 2
num_student_layers=$4 # 8

# Eval zero-shot
CUDA_VISIBLE_DEVICES=0 python offsite_tuning/eval_harness.py \
    --model_name_or_path $MODEL \
    --tasks ${TASK} \
    --output_dir logs/table1/${MODEL}/${TASK}/zeroshot.json &

# Eval ft all
CUDA_VISIBLE_DEVICES=1 python offsite_tuning/eval_harness.py \
    --model_name_or_path $MODEL \
    --tasks ${TASK} \
    --load_student logs/table1/${MODEL}/${TASK}/ft_all/student.pt \
    --output_dir logs/table1/${MODEL}/${TASK}/ft_all.json &

# Eval emulator zeroshot
CUDA_VISIBLE_DEVICES=2 python offsite_tuning/eval_harness.py \
    --model_name_or_path $MODEL \
    --tasks ${TASK} \
    --student_l_pad ${pad} \
    --student_r_pad ${pad} \
    --load_student logs/table1/${MODEL}/${TASK}/ft_distill_emulator/${num_student_layers}_${pad}_${pad}/student.pt \
    --output_dir logs/table1/${MODEL}/${TASK}/ft_distill_emulator/${num_student_layers}_${pad}_${pad}/emulator_zeroshot.json &

# Eval emulator ft
CUDA_VISIBLE_DEVICES=3 python offsite_tuning/eval_harness.py \
    --model_name_or_path $MODEL \
    --tasks ${TASK} \
    --student_l_pad ${pad} \
    --student_r_pad ${pad} \
    --load_adapter logs/table1/${MODEL}/${TASK}/ft_distill_emulator/${num_student_layers}_${pad}_${pad}/adapter.pt \
    --load_student logs/table1/${MODEL}/${TASK}/ft_distill_emulator/${num_student_layers}_${pad}_${pad}/student.pt \
    --output_dir logs/table1/${MODEL}/${TASK}/ft_distill_emulator/${num_student_layers}_${pad}_${pad}/emulator_ft.json &

# Eval plug-in
CUDA_VISIBLE_DEVICES=4 python offsite_tuning/eval_harness.py \
    --model_name_or_path $MODEL \
    --tasks ${TASK} \
    --student_l_pad ${pad} \
    --student_r_pad ${pad} \
    --load_adapter logs/table1/${MODEL}/${TASK}/ft_distill_emulator/${num_student_layers}_${pad}_${pad}/adapter.pt \
    --output_dir logs/table1/${MODEL}/${TASK}/ft_distill_emulator/${num_student_layers}_${pad}_${pad}/plugin.json &

wait
