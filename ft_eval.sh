python finetuning_eval.py --dataset_name BRCAM2C --subdataset_name mpp_0.25-size_$2-valid_224 --model_name $1 --eval_lr ${3:-1e-5} --eval_bs ${4:-32}

python finetuning_eval.py --dataset_name BRCAM2C --subdataset_name mpp_0.5-size_$2-valid_224 --model_name $1 --eval_lr ${3:-1e-5} --eval_bs ${4:-32}

python finetuning_eval.py --dataset_name OCELOT --subdataset_name mpp_0.25-size_$2-valid_224 --model_name $1 --eval_lr ${3:-1e-5} --eval_bs ${4:-32}

python finetuning_eval.py --dataset_name OCELOT --subdataset_name mpp_0.5-size_$2-valid_224 --model_name $1 --eval_lr ${3:-1e-5} --eval_bs ${4:-32}

python finetuning_eval.py --dataset_name PUMA --subdataset_name mpp_0.5-size_$2-label_coarse-valid_224 --model_name $1 --eval_lr ${3:-1e-5} --eval_bs ${4:-32}

python finetuning_eval.py --dataset_name PUMA --subdataset_name mpp_0.25-size_$2-label_coarse-valid_224 --model_name $1 --eval_lr ${3:-1e-5} --eval_bs ${4:-32}
