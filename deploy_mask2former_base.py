#!/bin/bash

# Define variables for the paths
DEPLOY_SCRIPT="../mmdeploy/tools/deploy.py"
DEPLOY_CONFIG="./tensorrt_static_512x1024_bs_1.py"
MODEL_CONFIG="./mmseg_mask2former_swinb_cityscapes_base"
CHECKPOINT="./mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024_20221203_045030-9a86a225.pth"
DEMO_IMG="../mmsegmentation/demo/demo.png"
WORK_DIR="./mask2former_cityscapes_engine"
DEVICE="cuda:0"
LOG_LEVEL="INFO"

# Run the command
python $DEPLOY_SCRIPT \
    $DEPLOY_CONFIG \
    $MODEL_CONFIG \
    $CHECKPOINT \
    $DEMO_IMG \
    --test-img $DEMO_IMG \
    --work-dir $WORK_DIR \
    --show \
    --device $DEVICE \
    --log-level $LOG_LEVEL
