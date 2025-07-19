DATA_DIR="../"
OUTPUT_DIR="./exp/model/fox_pro_760m_48b"  # You can set this to any other path
WANDB_DIR="./exp/wandb"  # You can set this to any other path
mkdir -p $OUTPUT_DIR
mkdir -p $WANDB_DIR
CUDA_VISIBLE_DEVICES=0,1 \
fabric run train.py \
    --devices 2 \
    --num-nodes 1 \
    --node-rank 0 \
    --main-address localhost \
    --main-port 1234 \
    +experiment/longcrawl64/forgetting_transformer=pro_760m_48b \
    seed=0 \
    exp=demo \
    tag=fox_pro_760m_48b \
    output_dir=$OUTPUT_DIR \
    data_dir=$DATA_DIR \
    wandb.log_dir=$WANDB_DIR \
    wandb.mode=offline \
    resume=true