TASK="arc_easy"
MODEL="gpt2-xl"
num_student_layers=16
bs=4
pad=2
eval_steps=10

lr=5e-5

bash scripts/table1/ft_all.sh ${TASK} ${MODEL} ${num_student_layers} ${bs} ${pad} ${eval_steps}

bash scripts/table1/ft_emulator.sh ${TASK} ${MODEL} ${num_student_layers} ${bs} ${pad} ${eval_steps} ${lr}

bash scripts/table1/eval.sh ${TASK} ${MODEL} ${pad} ${num_student_layers}

