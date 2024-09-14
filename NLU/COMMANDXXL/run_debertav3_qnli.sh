python ./NLU/examples/text-classification/run_glue.py \
--model_name_or_path "/home/dqxy/xyj/data/deberta-v2-xxlarge" \
--task_name qnli \
--apply_lora --apply_adalora --lora_type svd \
--target_rank 2  --lora_r 8   \
--reg_orth_coef 0.1 \
--init_warmup 8000 --final_warmup 32000 --mask_interval 100 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 32 \
--do_train --do_eval --max_seq_length 512 \
--per_device_train_batch_size 8 --learning_rate 5e-4  \
--num_train_epochs 5 --warmup_steps 2000 \
--cls_dropout 0.1 --weight_decay 0.01 \
--evaluation_strategy steps --eval_steps 4000 \
--save_strategy steps --save_steps 64000 \
--logging_steps 1200 \
--tb_writter_loginterval 1200 \
--report_to tensorboard \
--seed 6 \
--root_output_dir ./output/gluexxl/6qnli \
--overwrite_output_dir




