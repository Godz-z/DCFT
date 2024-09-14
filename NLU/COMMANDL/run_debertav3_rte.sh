python ./NLU/examples/text-classification/run_glue.py \
--model_name_or_path "/home/dqxy/xyj/data/RoBERTa-large" \
--task_name rte \
--apply_adalora --apply_lora \
--lora_type svd --target_rank 2  --lora_r 8  \
--reg_orth_coef 0.3 \
--init_warmup 600 --final_warmup 1800 --mask_interval 1 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 32 \
--do_train --do_eval --max_seq_length 320 \
--per_device_train_batch_size 32 --learning_rate 1.2e-3 \
--num_train_epochs 50 --warmup_steps 200 \
--cls_dropout 0.20 --weight_decay 0.01 \
--evaluation_strategy steps --eval_steps 100 \
--save_strategy steps --save_steps 10000 \
--logging_steps 10 --report_to tensorboard \
--seed 6 \
--root_output_dir ./output/glueL/6rte \
--overwrite_output_dir 

