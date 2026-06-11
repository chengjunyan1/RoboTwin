#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import os
from pathlib import Path

DEFAULT_ROBOTWIN_HF_REPO = "motus-robotics/pi0.5_robotwin2"
DEFAULT_ROBOTWIN_CONFIG = "pi05_base_finetune_on_robotwin_clean_randomized_joint_training"
DEFAULT_ROBOTWIN_MODEL = "pi0.5_robotwin2"


def _maybe_download_default_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    model_path = checkpoint_dir / "model.safetensors"
    assets_path = checkpoint_dir / "assets"
    if model_path.exists() and assets_path.is_dir():
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Default pi0.5 RoboTwin checkpoint is missing and huggingface_hub is not installed. "
            f"Install it or download {DEFAULT_ROBOTWIN_HF_REPO} into {checkpoint_dir}."
        ) from exc

    print(f"Downloading {DEFAULT_ROBOTWIN_HF_REPO} to {checkpoint_dir} ...")
    snapshot_download(
        repo_id=DEFAULT_ROBOTWIN_HF_REPO,
        local_dir=str(checkpoint_dir),
        allow_patterns=[
            "model.safetensors",
            "metadata.pt",
            "assets/**",
        ],
    )

class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        checkpoint_dir = f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}"
        if self.train_config_name == DEFAULT_ROBOTWIN_CONFIG and self.model_name == DEFAULT_ROBOTWIN_MODEL:
            _maybe_download_default_checkpoint(checkpoint_dir)

        specified_path = f"{checkpoint_dir}/assets/"
        if not os.path.isdir(specified_path):
            raise FileNotFoundError(
                f"Missing pi0.5 assets directory: {specified_path}. "
                f"Download {DEFAULT_ROBOTWIN_HF_REPO} into {checkpoint_dir}."
            )
        entries = os.listdir(specified_path)
        if not entries:
            raise FileNotFoundError(f"No asset id found under {specified_path}.")
        assets_id = entries[0]

        config = _config.get_config(self.train_config_name)
        self.policy = _policy_config.create_trained_policy(
            config,
            checkpoint_dir,
            robotwin_repo_id=assets_id,
            )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
