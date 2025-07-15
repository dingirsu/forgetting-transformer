source /gpfs/junlab/weijia24/miniconda3/bin/activate fox
DATA_DIR="/gpfs/junlab/weijia24/"
OUTPUT_DIR="/gpfs/junlab/weijia24/forgetting-transformer-sp/exp/fox_pro_360m_2_head"  # You can set this to any other path
WANDB_DIR="/gpfs/junlab/weijia24/forgetting-transformer-sp/exp/wandb"  # You can set this to any other path
mkdir -p $OUTPUT_DIR
mkdir -p $WANDB_DIR
fabric run train.py \
    --devices 8 \
    --num-nodes 1 \
    --node-rank 0 \
    --main-address localhost \
    --main-port 1235 \
    +experiment/longcrawl64/forgetting_transformer=pro_360m_7b \
    seed=0 \
    exp=demo \
    tag=fox_pro_360m_2_head \
    output_dir=$OUTPUT_DIR \
    data_dir=$DATA_DIR \
    wandb.log_dir=$WANDB_DIR \
    wandb.mode=offline \
    resume=true