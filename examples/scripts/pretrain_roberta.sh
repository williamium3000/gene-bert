cd examples

export KMER=6
export TRAIN_FILE=sample_data/pre/6_3k.txt
export TEST_FILE=sample_data/pre/6_3k.txt
export SOURCE=/home/mliu121/william/project/gene-bert/
export OUTPUT_PATH=work_dirs/roberta_test

python run_pretrain.py \
    --output_dir $OUTPUT_PATH \
    --model_type=roberta \
    --tokenizer_name=dna$KMER \
    --config_name=$SOURCE/src/transformers/dnabert-config/roberta-config-$KMER/config.json \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --gradient_accumulation_steps 32 \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 16 \
    --save_steps 500 \
    --mlm \
    --save_total_limit 20 \
    --max_steps 16000 \
    --evaluate_during_training \
    --logging_steps 500 \
    --line_by_line \
    --learning_rate 8e-3 \
    --block_size 512 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --beta1 0.9 \
    --beta2 0.98 \
    --mlm_probability 0.025 \
    --warmup_steps 10000 \
    --overwrite_output_dir \
    --n_process 2