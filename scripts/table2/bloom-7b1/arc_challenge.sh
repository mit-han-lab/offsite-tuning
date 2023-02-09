TASK="arc_challenge"
MODEL="bigscience/bloom-7b1"
num_student_layers=18
bs=1
pad=2

eval_steps=50
bash scripts/table2/ft_layerdrop.sh ${TASK} ${MODEL} ${num_student_layers} ${bs} ${pad} ${eval_steps} ${lr}

bash scripts/table2/eval_layerdrop.sh ${TASK} ${MODEL} ${pad} ${num_student_layers}
