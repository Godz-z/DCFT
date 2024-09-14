python ./NLU/examples/text-classification/run_glue.py \
--model_name_or_path "/home/dqxy/xyj/data/RoBERTa-large" \
--task_name mnli \
--apply_adalora --apply_lora --lora_type svd \
--target_rank 2  --lora_r 8  \
--reg_orth_coef 0.1 \
--init_warmup 8000 --final_warmup 50000 --mask_interval 100 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 16 \
--do_train --do_eval \
--max_seq_length 256 \
--per_device_train_batch_size 32 --learning_rate 5e-4 --num_train_epochs 7 \
--warmup_steps 1000 \
--cls_dropout 0.15 --weight_decay 0 \
--evaluation_strategy steps --eval_steps 3000 \
--save_strategy steps --save_steps 30000 \
--logging_steps 500 \
--seed 6 \
--root_output_dir ./output/glueL/6mnli \
--overwrite_output_dir
