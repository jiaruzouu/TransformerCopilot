# bin bash
MODEL=(
    "google/flan-t5-small"
    "google/flan-t5-large"
    "google/flan-t5-base"
)
EPOCH=0
# Define the test set list
TEST_SETS=(
    "mawps"
    "svamp" 
    "gsm8k" # Add more test sets here
)

# Path to the training set
TRAIN_SET=../ft-training_set/math_10k.json

PORT=2000
# Loop through each model in the MODEL array
for BACKBONE in "${MODEL[@]}"; do
    echo "Running training and testing with model: $BACKBONE"

    # Set batch size conditionally
    if [[ "$BACKBONE" == "google/flan-t5-large" ]]; then
        BATCH_SIZE=8
        EPOCH=10
    else
        BATCH_SIZE=16
        EPOCH=10
    fi

    # Loop through each test set in TEST_SETS
    for TEST_SET in "${TEST_SETS[@]}"; do
        echo "Testing with test set: $TEST_SET"

        python ../src/src_t5/main.py \
            --generation_train_file $TRAIN_SET \
            --generation_test_file ../data/$TEST_SET/test.json \
            --backbone $BACKBONE \
            --datasets $TEST_SET \
            --distributed 1 \
            --gpu 0,1 \
            --tasks Generation \
            --item_indexing sequential \
            --epochs $EPOCH \
            --batch_size $BATCH_SIZE \
            --master_port $PORT \
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
            --test_filtered_batch 0 \

        PORT=$((PORT + 1))
    done
done
