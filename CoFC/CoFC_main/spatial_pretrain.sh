#!/bin/bash
cd $(dirname $0)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source ~/.bashrc
source "$SCRIPT_DIR/utils.sh"
mkdir pretrain_models/spa
mkdir pretrain_models/ene
root_path="../../../.."
export PYTHONPATH="$(pwd)/$root_path/":$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
encoder_config="model_configs/gnn_settings.json"
model_config="model_configs/pretrain.json"
data_path="'./pretrain_catch_data'"
python spatial_pretrain.py \
		 --batch_size=256 \
		--num_workers=4 \
		--max_epoch=50 \
		--lr=1e-3 \
		--dropout_rate=0.2 \
		--data_path=$data_path \
		--encoder_config=$encoder_config \
		--model_config=$model_config \
