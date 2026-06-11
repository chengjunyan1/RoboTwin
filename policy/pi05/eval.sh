#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G

policy_name=pi05
default_train_config_name=pi05_base_finetune_on_robotwin_clean_randomized_joint_training
default_model_name=pi0.5_robotwin2
default_checkpoint_id=30000
default_pi0_step=32

task_name=${1}
task_config=${2}
train_config_name=${3:-$default_train_config_name}
model_name=${4:-$default_model_name}
seed=${5:-0}
gpu_id=${6:-0}

if [ -z "$task_name" ] || [ -z "$task_config" ]; then
    echo "Usage: bash eval.sh <task_name> <task_config> [train_config_name] [model_name] [seed] [gpu_id]"
    echo "Defaults: train_config_name=$default_train_config_name, model_name=$default_model_name"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# source .venv/bin/activate
cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --checkpoint_id ${default_checkpoint_id} \
    --pi0_step ${default_pi0_step} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} 
