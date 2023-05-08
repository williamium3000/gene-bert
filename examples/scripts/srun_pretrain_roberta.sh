cd examples

export KMER=6
export TRAIN_FILE=sample_data/pre/6_3k.txt
export TEST_FILE=sample_data/pre/6_3k.txt
export SOURCE=/home/mliu121/william/project/gene-bert/
export OUTPUT_PATH=work_dirs/roberta_6_3k$KMER

NOW=$(date +"%Y%m%d_%H%M%S")
mkdir -p $OUTPUT_PATH

srun --partition ica100 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --gres=gpu:1 \
    --job-name=roberta \
    --mem=50g \
    --time 02-00:00:00 \
    -A danielk80_gpu \
    python run_pretrain.py \
    --output_dir $OUTPUT_PATH \
    --model_type=roberta \
    --tokenizer_name=dna$KMER \
    --config_name=$SOURCE/src/transformers/dnabert-config/roberta-config-$KMER/config.json \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --mlm \
    --eval_data_file=$TEST_FILE \
    --gradient_accumulation_steps 25 \
    --per_gpu_train_batch_size 10 \
    --per_gpu_eval_batch_size 6 \
    --save_steps 500 \
    --save_total_limit 20 \
    --max_steps 200000 \
    --evaluate_during_training \
    --logging_steps 500 \
    --line_by_line \
    --learning_rate 4e-4 \
    --block_size 512 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --beta1 0.9 \
    --beta2 0.98 \
    --mlm_probability 0.025 \
    --warmup_steps 10000 \
    --overwrite_output_dir \
    --n_process 24 2>&1 | tee $OUTPUT_PATH/$NOW.txt &