# bin bash 
MODEL=(
    "google/flan-t5-small"
    "google/flan-t5-base"
    "google/flan-t5-large"
)

# Paths to training and testing sets
TRAIN_SET=../ft-training_set/commonsense_120k.json
TEST_SET=aqua # change to other test sets

# Loop through each model in the MODEL array
for BACKBONE in "${MODEL[@]}"; do
    echo "Running training and testing with model: $BACKBONE"
    
    python ../src/src_t5/main.py \
        --generation_train_file $TRAIN_SET \
        --generation_test_file ../data/$TEST_SET/test.json \
        --backbone $BACKBONE \
        --datasets $TEST_SET \
        --distributed 1 \
        --gpu 0,3 \
        --tasks Generation \
        --item_indexing sequential \
        --epochs 20 \
        --batch_size 2 \
        --master_port 1999 \
        --prompt_file ../prompt.txt \
        --sample_prompt 1 \
        --eval_batch_size 1 \
        --dist_sampler 0 \
        --max_his 20 \
        --sample_num 3,3 \
        --train 1 \
        --test_prompt seen:0 \
        --lr 1e-3 \
        --test_before_train 0 \
        --test_epoch 0 \
        --collaborative_token_size 500 \
        --collaborative_cluster 20 \
        --collaborative_sparse 0 \
        --collaborative_float32 0 \
        --test_filtered 0 \
        --test_filtered_batch 0
done
