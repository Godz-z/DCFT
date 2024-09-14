python ./NLU/examples/text-classification/run_glue.py \
--model_name_or_path "/home/dqxy/xyj/data/deberta-v2-xxlarge" \
--task_name cola \
--apply_lora --apply_adalora --lora_type svd \
--target_rank 2   --lora_r 8   \
--reg_orth_coef 0.1 \
--init_warmup 6400 --final_warmup 28000 --mask_interval 10 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 32 \
--do_train --do_eval --max_seq_length 64 \
--per_device_train_batch_size 4 --learning_rate 8e-4 \
--num_train_epochs 25 --warmup_steps 800 \
--cls_dropout 0.10 --weight_decay 0.00 \
--evaluation_strategy steps --eval_steps 800 \
--save_strategy steps --save_steps 80000 \
--logging_steps 80 \
--tb_writter_loginterval 100 \
--report_to tensorboard \
--seed 6 \
--root_output_dir ./output/gluexxl/6cola \
--overwrite_output_dir


