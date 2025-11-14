# bash bin 
TRAIN_DATASET=commonsense_170k
METHOD_TYPE=explore
TEST_DATASET=piqa
MODEL=meta-llama/Llama-3.1-8B
EXPLORE_MODEL="meta-llama/Llama-3.2-1B"
NUM_EPOCHS=3
ADAPTER=lora
EXPLORE_LOGITS_FACTOR=100
export TOKENIZERS_PARALLELISM=false

mkdir -p ./trained_models/$METHOD_TYPE/$TRAIN_DATASET/$NUM_EPOCHS/${MODEL}-${ADAPTER}/
mkdir -p ./trained_models/$METHOD_TYPE/$TRAIN_DATASET/$NUM_EPOCHS/${MODEL}-${ADAPTER}-Copilot/
mkdir -p ./experiment/meta-llama/
mkdir -p ./logs/$TRAIN_DATASET/$NUM_EPOCHS/$METHOD_TYPE/meta-llama/

CUDA_VISIBLE_DEVICES=6 python finetune.py\
  --base_model  $MODEL \
  --explore_base_model $EXPLORE_MODEL \
  --data_path  ./ft-training_set/${TRAIN_DATASET}.json \
  --test_data_path  ./dataset/$TEST_DATASET/test.json \
  --output_dir ./trained_models/$METHOD_TYPE/$TRAIN_DATASET/$NUM_EPOCHS/${MODEL}-${ADAPTER}/ \
  --explore_output_dir ./trained_models/$METHOD_TYPE/$TRAIN_DATASET/$NUM_EPOCHS/${MODEL}-${ADAPTER}-Copilot/ \
  --batch_size 16 \
  --micro_batch_size 4 \
  --num_epochs $NUM_EPOCHS \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --eval_step 80 \
  --save_step 80 \
  --adapter_name $ADAPTER \
  --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
  --lora_r 32 \
  --lora_alpha 64 \
  --dataset_name $DATASET \
  --explore_flag False \
  --explore_logits_factor $EXPLORE_LOGITS_FACTOR \
  --direct_test False \


TEST_DATASETS=("piqa" "hellaswag" "openbookqa" "winogrande" "social_i_qa" "boolq") 
DECODING=greedy
TOP_K=-1
export TOKENIZERS_PARALLELISM=false

for TEST_DATASET in "${TEST_DATASETS[@]}"; do
    LOG_DIR="logs/$TRAIN_DATASET/$NUM_EPOCHS/$METHOD_TYPE"
    LOG_FILE="$LOG_DIR/${MODEL}-${ADAPTER}-${TEST_DATASET}-${METHOD_TYPE}-${DECODING}-Copilot.log"
    
    mkdir -p "$LOG_DIR"

    echo "Evaluating on dataset: $TEST_DATASET" | tee -a "$LOG_FILE"
    
    CUDA_VISIBLE_DEVICES=3 python commonsense_evaluate.py \
        --model $MODEL \
        --adapter $ADAPTER \
        --dataset $TEST_DATASET \
        --base_model $MODEL \
        --batch_size 1 \
        --lora_weights ./trained_models/$METHOD_TYPE/$TRAIN_DATASET/$NUM_EPOCHS/${MODEL}-${ADAPTER}/ \
        --explore_model $EXPLORE_MODEL \
        --base_explore_model $EXPLORE_MODEL \
        --explore_lora_weights ./trained_models/$METHOD_TYPE/$TRAIN_DATASET/$NUM_EPOCHS/${MODEL}-${ADAPTER}-Copilot/ \
        --explore_flag 1 \
        --explore_weight 0.1 \
        --decoding_method $DECODING \
        --explore_logits_factor $EXPLORE_LOGITS_FACTOR\
        --top_k_explore_logits $TOP_K \
        --logfile $LOG_FILE \

done