#!/bin/bash
set -euxo pipefail

# Read distributed environment variables
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NNODES=${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}
NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-1234}}

# NCCL config
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=12
export NCCL_P2P_DISABLE=0
export NCCL_CUMEM_ENABLE=0

export PYTHONPATH=$WORKING_DIR

# Ray specific settings
RAY_PORT=6379
RAY_HEAD_IP="$MASTER_ADDR:$RAY_PORT"
# Number of seconds to wait for the head node to be ready.
WAIT_INTERVAL_HEAD=30
# Number of seconds to wait for cluster to be ready before submitting the job.
WAIT_INTERVAL_SUBMIT=60

# Start Ray processes based on node rank
if [ "$NODE_RANK" -eq 0 ]; then
    echo "Starting HEAD node at $MASTER_ADDR"
    ray start --head \
        --node-ip-address=$MASTER_ADDR \
        --port=$RAY_PORT \
        --num-cpus=$NPROC_PER_NODE \
        --block &

    # Wait for Ray head node to be ready
    sleep $WAIT_INTERVAL_SUBMIT

    # Submit the Ray job from the head node
    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json="{\"working_dir\": \"${WORKING_DIR}\"}" \
        -- python3 -m openrlhf.cli.train_ppo_ray \
        --ref_num_nodes 4 \
        --ref_num_gpus_per_node 8 \
        --critic_num_nodes 4 \
        --critic_num_gpus_per_node 8 \
        --actor_num_nodes 4 \
        --actor_num_gpus_per_node 8 \
        --vllm_num_engines 64 \
        --vllm_tensor_parallel_size 1 \
        --colocate_actor_ref \
        --pretrain /redacted/path/here\
        --remote_rm_url math_rule_cosine_scale,math_rule_repetition_dense \
        --math_rule_common_exceed_length -10 \
        --math_rule_common_repetition_ngram_size 40 \
        --math_rule_repetition_dense_penalty -0.05 \
        --math_rule_cosine_scale_min_value_wrong -10 \
        --math_rule_cosine_scale_max_value_wrong 0 \
        --math_rule_cosine_scale_min_value_correct 1 \
        --math_rule_cosine_scale_max_value_correct 2 \
        --save_path /redacted/path/here\
        --ckpt_path /redacted/path/here\
        --load_checkpoint \
        --save_steps 8 \
        --max_ckpt_num 9999999 \
        --max_ckpt_mem 9999999 \
        --micro_train_batch_size 1 \
        --train_batch_size 512 \
        --micro_rollout_batch_size 1 \
        --rollout_batch_size 512 \
        --n_samples_per_prompt 8 \
        --max_samples 100000 \
        --max_epochs 1 \
        --num_episodes 8 \
        --lambd 1 \
        --gamma 1,0.99 \
        --prompt_max_len 2048 \
        --generate_max_len 14336 \
        --zero_stage 2 \
        --bf16 \
        --actor_learning_rate 5e-7 \
        --critic_learning_rate 9e-6 \
        --init_kl_coef 0.01 \
        --prompt_data /redacted/path/here\
        --input_key prompt \
        --answer_key answer \
        --apply_chat_template \
        --normalize_reward \
        --flash_attn \
        --gradient_checkpointing \
        --use_wandb <redacted key> \
        --wandb_project cot \
        --wandb_run_name "exp_ppo_cosine_qwq_sp8_ep8_bs512_epoch1_w10_c2_ngram40_pen0d05_gg100_99_16k_$(date "+%Y%m%d_%H%M%S")"
else
    echo "Starting WORKER node $NODE_RANK"
    sleep $WAIT_INTERVAL_HEAD
    ray start --address "$RAY_HEAD_IP" \
        --num-cpus=$NPROC_PER_NODE \
        --block
fi