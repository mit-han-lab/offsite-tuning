TASK="web_questions"
MODEL="gpt2-xl"
num_student_layers=16
bs=4
pad=2
eval_steps=100

lr=1e-4

bash scripts/table1/ft_all.sh ${TASK} ${MODEL} ${num_student_layers} ${bs} ${pad} ${eval_steps}

bash scripts/table1/ft_emulator.sh ${TASK} ${MODEL} ${num_student_layers} ${bs} ${pad} ${eval_steps} ${lr}

bash scripts/table1/eval.sh ${TASK} ${MODEL} ${pad} ${num_student_layers}

