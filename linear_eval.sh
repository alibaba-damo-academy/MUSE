python linear_knn_eval.py --eval_type linear --dataset_name BRCAM2C --subdataset_name mpp_0.25-size_$2-valid_224 --model_name $1

python linear_knn_eval.py --eval_type linear --dataset_name BRCAM2C --subdataset_name mpp_0.5-size_$2-valid_224 --model_name $1

python linear_knn_eval.py --eval_type linear --dataset_name OCELOT --subdataset_name mpp_0.25-size_$2-valid_224 --model_name $1

python linear_knn_eval.py --eval_type linear --dataset_name OCELOT --subdataset_name mpp_0.5-size_$2-valid_224 --model_name $1

python linear_knn_eval.py --eval_type linear --dataset_name PUMA --subdataset_name mpp_0.5-size_$2-label_coarse-valid_224 --model_name $1

python linear_knn_eval.py --eval_type linear --dataset_name PUMA --subdataset_name mpp_0.25-size_$2-label_coarse-valid_224 --model_name $1