# pi0.5 RoboTwin Evaluation

This folder defaults to the Hugging Face checkpoint `motus-robotics/pi0.5_robotwin2`.

Run evaluation from this directory:

```bash
cd extern/RoboTwin/policy/pi05
bash eval.sh <task_name> <task_config> [train_config_name] [model_name] [seed] [gpu_id]
```

If `train_config_name` and `model_name` are omitted, `eval.sh` uses:

```text
train_config_name=pi05_base_finetune_on_robotwin_clean_randomized_joint_training
model_name=pi0.5_robotwin2
checkpoint_id=30000
pi0_step=32
```

On first use, `pi_model.py` downloads the evaluation files from Hugging Face into:

```text
policy/pi05/checkpoints/pi05_base_finetune_on_robotwin_clean_randomized_joint_training/pi0.5_robotwin2/30000
```

Only `model.safetensors`, `metadata.pt`, and `assets/**` are downloaded. The large `optimizer.pt` file is not needed for evaluation.

To download manually instead:

```bash
cd extern/RoboTwin
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="motus-robotics/pi0.5_robotwin2",
    local_dir="policy/pi05/checkpoints/pi05_base_finetune_on_robotwin_clean_randomized_joint_training/pi0.5_robotwin2/30000",
    allow_patterns=["model.safetensors", "metadata.pt", "assets/**"],
)
PY
```
