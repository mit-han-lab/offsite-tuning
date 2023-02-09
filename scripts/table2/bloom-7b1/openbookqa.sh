TASK="openbookqa"
MODEL="bigscience/bloom-7b1"
num_student_layers=18
bs=1
pad=2
eval_steps=200

bash scripts/table2/ft_layerdrop.sh ${TASK} ${MODEL} ${num_student_layers} ${bs} ${pad} ${eval_steps} ${lr}

bash scripts/nlp_tasks/eval_layerdrop.sh ${TASK} ${FAMILY} ${MODEL} ${pad} ${num_student_layers}