#!/bin/bash -l
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
TRAIN_SCRIPT=$SCRIPT_DIR/../train.sh
CONFIGS=(
    "docclip_ViT-B-16-plus-240+laion400m_e31 $TRAIN_SCRIPT ViT-B-16-plus-240 laion400m_e31 256"
    "docclip_ViT-L-14+commonpool_xl_s13b_b90k $TRAIN_SCRIPT ViT-L-14 commonpool_xl_s13b_b90k 128"
)
# $TRAIN_SCRIPT -c doc_privacy_dp --type dp +experiment=dp_train/rvlcdip/alexnet lr=0.05 bs=2048 max_grad_norm=10.0 experiment_name=dp_train
# $TRAIN_SCRIPT -c doc_privacy_dp --type dp +experiment=dp_train/rvlcdip/resnet50 lr=0.05 bs=2048 max_grad_norm=10.0 experiment_name=dp_train
# $TRAIN_SCRIPT -c doc_privacy_dp --type dp +experiment=dp_train/rvlcdip/vgg16 lr=0.05 bs=2048 max_grad_norm=10.0 experiment_name=dp_train
# $TRAIN_SCRIPT -c doc_privacy_dp --type dp +experiment=dp_train/rvlcdip/efficientnet_b4 lr=0.05 bs=2048 max_grad_norm=10.0 experiment_name=dp_train
# $TRAIN_SCRIPT -c doc_privacy_dp --type dp +experiment=dp_train/rvlcdip/convnext_base lr=0.05 bs=2048 max_grad_norm=10.0 experiment_name=dp_train
# $TRAIN_SCRIPT -c doc_privacy_dp --type dp +experiment=dp_train/rvlcdip/docxclassifier_base lr=0.05 bs=2048 max_grad_norm=10.0 experiment_name=dp_train