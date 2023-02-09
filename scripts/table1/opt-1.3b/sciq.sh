TASK="sciq"
MODEL="facebook/opt-1.3b"
num_student_layers=8
bs=4
pad=2
eval_steps=100

lr=5e-5

bash scripts/table1/ft_all.sh ${TASK} ${MODEL} ${num_student_layers} ${bs} ${pad} ${eval_steps}

bash scripts/table1/ft_emulator.sh ${TASK} ${MODEL} ${num_student_layers} ${bs} ${pad} ${eval_steps} ${lr}

bash scripts/table1/eval.sh ${TASK} ${MODEL} ${pad} ${num_student_layers}

