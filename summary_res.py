"""
This file is used to summary the results of linear and knn evaluation.
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--eval_size', type=int, default=224)
args = parser.parse_args()

models = [args.model_name]

tasks = [
    f'BRCAM2C_mpp_0.5-size_{args.eval_size}-valid_224_knn_2025',
    f'BRCAM2C_mpp_0.5-size_{args.eval_size}-valid_224_linear_2025',
    f'BRCAM2C_mpp_0.5-size_{args.eval_size}-valid_224_finetuning_2025',
    f'OCELOT_mpp_0.5-size_{args.eval_size}-valid_224_knn_2025',
    f'OCELOT_mpp_0.5-size_{args.eval_size}-valid_224_linear_2025',
    f'OCELOT_mpp_0.5-size_{args.eval_size}-valid_224_finetuning_2025',
    f'PUMA_mpp_0.5-size_{args.eval_size}-label_coarse-valid_224_knn_2025',
    f'PUMA_mpp_0.5-size_{args.eval_size}-label_coarse-valid_224_linear_2025',
    f'PUMA_mpp_0.5-size_{args.eval_size}-label_coarse-valid_224_finetuning_2025',
    f'BRCAM2C_mpp_0.25-size_{args.eval_size}-valid_224_knn_2025',
    f'BRCAM2C_mpp_0.25-size_{args.eval_size}-valid_224_linear_2025',
    f'BRCAM2C_mpp_0.25-size_{args.eval_size}-valid_224_finetuning_2025',
    f'OCELOT_mpp_0.25-size_{args.eval_size}-valid_224_knn_2025',
    f'OCELOT_mpp_0.25-size_{args.eval_size}-valid_224_linear_2025',
    f'OCELOT_mpp_0.25-size_{args.eval_size}-valid_224_finetuning_2025',
    f'PUMA_mpp_0.25-size_{args.eval_size}-label_coarse-valid_224_knn_2025',
    f'PUMA_mpp_0.25-size_{args.eval_size}-label_coarse-valid_224_linear_2025',
    f'PUMA_mpp_0.25-size_{args.eval_size}-label_coarse-valid_224_finetuning_2025',
]

res_folder = './outputs'

for model_name in models:

    res_str = ''

    overall_acc_linear = 0
    overall_acc_knn = 0
    overall_acc_ft = 0
    for task_name in tasks:
        if 'linear' in task_name:
            log_file = f'{res_folder}/{model_name}_{task_name}/eval_linear.txt'
            with open(log_file, 'r') as f:
                # read the last line
                last_line = f.readlines()[-1]
                test_acc = float(last_line.split(': ')[-1]) * 100
            overall_acc_linear += test_acc
        elif 'knn' in task_name:
            log_file = f'{res_folder}/{model_name}_{task_name}/eval_knn.txt'
            with open(log_file, 'r') as f:
                # read the last line
                last_line = f.readlines()[-1]
                test_acc = float(last_line.split(': ')[-1])
            overall_acc_knn += test_acc
        elif 'finetuning' in task_name:
            log_file = f'{res_folder}/{model_name}_{task_name}/eval_linear.txt'
            with open(log_file, 'r') as f:
                # read the last line
                last_line = f.readlines()[-1]
                test_acc = float(last_line.split(': ')[-1]) * 100
            overall_acc_ft += test_acc
        else:
            raise ValueError('Unknown task name')
        res_str += ' & {:.2f}'.format(test_acc)
        # overall_acc += test_acc
    
    overall_acc_linear /= (len(tasks) // 3)
    overall_acc_knn /= (len(tasks) // 3)
    overall_acc_ft /= (len(tasks) // 3)
    res_str += ' & {:.2f} & {:.2f} & {:.2f}'.format(overall_acc_knn, overall_acc_linear, overall_acc_ft)

    print(model_name, res_str)
