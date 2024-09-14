python -m torch.distributed.launch --master_port=8679 --nproc_per_node=1 \
examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name stsb \
--apply_lora --apply_adalora --lora_type svd \
--target_rank 1  --lora_r 2    \
--reg_orth_coef 0.3 \
--init_warmup 800 --final_warmup 2000 --mask_interval 10 \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 32 \
--do_train --do_eval --max_seq_length 128 \
--per_device_train_batch_size 32 --learning_rate 2.2e-3 \
--num_train_epochs 25 --warmup_steps 100 \
--cls_dropout 0.2 --weight_decay 0.1 \
--evaluation_strategy steps --eval_steps 100 \
--save_strategy steps --save_steps 10000 \
--logging_steps 50 \
--tb_writter_loginterval 50 \
--report_to tensorboard \
--seed 6 \
--root_output_dir ./output/debertav3-base/stsb \
--overwrite_output_dir