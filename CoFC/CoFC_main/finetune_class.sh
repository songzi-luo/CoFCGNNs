#!/bin/bash
cd $(dirname $0)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

source ~/.bashrc
source "$SCRIPT_DIR/utils.sh"

root_path="$(pwd)/../../../.."
export PYTHONPATH="$root_path/":$PYTHONPATH

datasets="bace"
compound_encoder_config="model_configs/gnn_l8.json"
init_model="/root/hh/CoFC-main/CoFC/pretrain_models/ene/epoch23.pdparams"
log_prefix="log/prete"
thread_num=4
count=0

for dataset in $datasets; do
	echo "==> $dataset"
	data_path="downstream_datasets"
	cached_data_path="cached_data"
	if [ ! -f "$dataset.done" ]; then
		python finetune_class.py \
				--task=data \
				--num_workers=10 \
				--dataset_name=$dataset \
				--data_path=$data_path \
				--cached_data_path=$cached_data_path \
				--encoder_config=$compound_encoder_config \
				--model_config="model_configs/down_mlp3.json"
		if [ $? -ne 0 ]; then
			echo "Generate data failed for $dataset"
			exit 1
		fi
		touch $dataset.done
	fi

	model_config_list="model_configs/down_mlp2.json model_configs/down_mlp3.json"
	lrs_list="1e-3,1e-3 4e-3,4e-3 1e-4,1e-3"
	drop_list="0.2 0.5"
	if [ "$dataset" == "tox21" ] || [ "$dataset" == "toxcast" ]; then
		batch_size=128
  elif [ "$dataset" == "bace" ] || [ "$dataset" == "bbbp" ] || [ "$dataset" == "clintox" ] || [ "$dataset" == "sider" ]; then
    batch_size=32
	else
		batch_size=256
	fi
	for model_config in $model_config_list; do
		for lrs in $lrs_list; do
			IFS=, read -r -a array <<< "$lrs"
			lr=${array[0]}
			head_lr=${array[1]}
			for dropout_rate in $drop_list; do
				log_dir="$log_prefix-$dataset"
				log_file="$log_dir/lr${lr}_${head_lr}-drop${dropout_rate}.txt"
				echo "Outputs redirected to $log_file"
				mkdir -p $log_dir
				for time in $(seq 1 4); do
					{
						gpu_index=$((time-1))  
						python -m paddle.distributed.launch --gpus $gpu_index finetune_class.py \
								--batch_size=$batch_size \
								--dataset_name=$dataset \
								--data_path=$data_path \
								--cached_data_path=$cached_data_path \
								--split_type=scaffold \
								--encoder_config=$compound_encoder_config \
								--model_config=$model_config \
								--init_model=$init_model \
								--model_dir=./finetune_models/$dataset \
								--encoder_lr=$lr \
								--head_lr=$head_lr \
								--dropout_rate=$dropout_rate >> $log_file 2>&1

						cat $log_dir/* | grep FINAL| python ana_results.py > $log_dir/final_result
					} &
					let count+=1
					if [[ $(($count % $thread_num)) -eq 0 ]]; then
						wait
					fi
				done
			done
		done
	done
done
wait

