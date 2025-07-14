OUTPUT_DIR=${1:-"./alma-7b-parallel-ft-lora"}
pairs=${2:-"de-en,cs-en,is-en,zh-en,ru-en,en-de,en-cs,en-is,en-zh,en-ru"}
LORA_RANK=${3:-"16"}

CUDA_VISIBLE_DEVICES=0 python3 run_llmmt.py \
    --model_name_or_path haoranxu/ALMA-7B-R \
    --mmt_data_path  ./training_data/remapped_onlytags \
    --use_peft \
    --lora_rank ${LORA_RANK} \
    --do_train \
    --do_eval \
    --eval_strategy steps \
    --do_predict \
    --language_pairs ${pairs} \
    --load_best_model_at_end \
    --low_cpu_mem_usage \
    --bf16 true \
    --learning_rate 2e-3 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_ratio 0.01 \
    --ignore_pad_token_for_loss \
    --ignore_prompt_token_for_loss \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --eval_steps 0.05 \
    --save_strategy steps \
    --save_steps 0.1 \
    --save_total_limit 1 \
    --logging_strategy steps \
    --logging_steps 0.05 \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --predict_with_generate \
    --prediction_loss_only \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --seed 42 \
    --overwrite_output_dir \
    --num_beams 5 \
    --ddp_timeout 999999 \
    --report_to wandb \
    --overwrite_cache
