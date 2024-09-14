python ./NLU/examples/text-classification/run_glue.py \
--model_name_or_path "/home/dqxy/xyj/data/deberta-v2-xxlarge" \
--task_name mrpc \
--apply_lora --apply_adalora --lora_type svd \
--target_rank 2   --lora_r 8   \
--reg_orth_coef 0.1 \
--init_warmup 1200 --final_warmup 3600 --mask_interval 1 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 32 \
--do_train --do_eval --max_seq_length 320 \
--per_device_train_batch_size 16 --learning_rate 1e-3 \
--num_train_epochs 30 --warmup_ratio 0.1 \
--cls_dropout 0.0 --weight_decay 0.01 \
--evaluation_strategy steps --eval_steps 600 \
--save_strategy steps --save_steps 6000 \
--logging_steps 200 \
--report_to tensorboard \
--seed 6 \
--root_output_dir ./output/gluexxl/6mrpc \
--overwrite_output_dir



