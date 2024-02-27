import os
import re
import matplotlib.pyplot as plt

import numpy as np

def read_log_file(file_path):
    total_steps = []
    p_values = []
    v_values = []
    q1_values = []
    q2_values = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines, 1): 
            if line.startswith('total steps:'):
                total_steps.append(int(re.findall(r'\d+', line)[0]))
            elif line.startswith('gaussian_policy'):
                p_data = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                if len(p_data) != 3: 
                    print(f"Error in line {idx}: {line.strip()}",p_data) 
                else:
                    p_values.append((float(p_data[0]), float(p_data[1]), float(p_data[2])))
            elif line.startswith('v'):
                v_data = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                if len(v_data) != 3: 
                    print(f"Error in line {idx}: {line.strip()}",v_data) 
                else:
                    v_values.append((float(v_data[0]), float(v_data[1]), float(v_data[2])))
            elif line.startswith('q1'):
                q1_data = re.findall(r'[-+]?\d*\.\d+|\d+', line[2:])
                if len(q1_data) != 3: 
                    print(f"Error in line {idx}: {line.strip()}",q1_data) 
                else:
                    q1_values.append((float(q1_data[0]), float(q1_data[1]), float(q1_data[2])))
            elif line.startswith('q2'):
                q2_data = re.findall(r'[-+]?\d*\.\d+|\d+', line[2:])
                if len(q2_data) != 3: 
                    print(f"Error in line {idx}: {line.strip()}",q2_data) 
                else:
                    q2_values.append((float(q2_data[0]), float(q2_data[1]), float(q2_data[2])))
    return total_steps, p_values, v_values, q1_values, q2_values

# 指定log文件所在的文件夹路径
log_folder_path = "/home/lzy/work/CORL/algorithms/offline/"  
save_folder_path = "/home/lzy/work/CORL/algorithms/pics"  

p_data_feature_rank = {}
p_data_weight_norm = {}
p_data_fau = {}
v_data_feature_rank = {}
v_data_weight_norm = {}
v_data_fau = {}
q1_data_feature_rank = {}
q1_data_weight_norm = {}
q1_data_fau = {}
q2_data_feature_rank = {}
q2_data_weight_norm = {}
q2_data_fau = {}
total_steps = None
for file_name in os.listdir(log_folder_path):
    if file_name.endswith(".log"):
        file_path = os.path.join(log_folder_path, file_name)
        total_steps, p_values, v_values, q1_values, q2_values = read_log_file(file_path)
        algo, seed = file_name[:-4].split('_')[0], file_name[:-4].split('_')[-1]
        env = file_name[4:-6]
        key = f"{algo}_{env}"
        if key in p_data_feature_rank:
            p_data_feature_rank[key].append([q2[0] for q2 in p_values])
            p_data_weight_norm[key].append([q2[1] for q2 in p_values])
            p_data_fau[key].append([q2[2] for q2 in p_values])
            v_data_feature_rank[key].append([q2[0] for q2 in v_values])
            v_data_weight_norm[key].append([q2[1] for q2 in v_values])
            v_data_fau[key].append([q2[2] for q2 in v_values])
            q1_data_feature_rank[key].append([q2[0] for q2 in q1_values])
            q1_data_weight_norm[key].append([q2[1] for q2 in q1_values])
            q1_data_fau[key].append([q2[2] for q2 in q1_values])
            q2_data_feature_rank[key].append([q2[0] for q2 in q2_values])
            q2_data_weight_norm[key].append([q2[1] for q2 in q2_values])
            q2_data_fau[key].append([q2[2] for q2 in q2_values])
        else:
            p_data_feature_rank[key] = [[q2[0] for q2 in p_values]]
            p_data_weight_norm[key] = [[q2[1] for q2 in p_values]]
            p_data_fau[key] = [[q2[2] for q2 in p_values]]
            v_data_feature_rank[key] = [[q2[0] for q2 in v_values]]
            v_data_weight_norm[key] = [[q2[1] for q2 in v_values]]
            v_data_fau[key] = [[q2[2] for q2 in v_values]]
            q1_data_feature_rank[key] = [[q2[0] for q2 in q1_values]]
            q1_data_weight_norm[key] = [[q2[1] for q2 in q1_values]]
            q1_data_fau[key] = [[q2[2] for q2 in q1_values]]
            q2_data_feature_rank[key] = [[q2[0] for q2 in q2_values]]
            q2_data_weight_norm[key] = [[q2[1] for q2 in q2_values]]
            q2_data_fau[key] = [[q2[2] for q2 in q2_values]]

name_map = {
    0: "policy",
    1: "v_function",
    2: "critic_1",
    3: "critic_2"
}

idx_map = {
    0: "p",
    1: "v",
    2: "q1",
    3: "q2"
}

for idx, (data_feature_rank, data_weight_norm, data_fau) in enumerate([(p_data_feature_rank, p_data_weight_norm, p_data_fau), (v_data_feature_rank, v_data_weight_norm, v_data_fau), (q1_data_feature_rank, q1_data_weight_norm, q1_data_fau), (q2_data_feature_rank, q2_data_weight_norm, q2_data_fau)]):

    #print(idx)
    #print(len(data_feature_rank))

    for x in data_feature_rank:
        #feature_rank_mean = list(np.array(data_feature_rank[x]).mean(axis=0))
        #feature_rank_std = list(np.array(data_feature_rank[x]).var(axis=0))
        plt.figure(figsize=(10, 6))
        feature_rank_mean = np.array(data_feature_rank[x]).mean(axis=0)
        feature_rank_std = np.array(data_feature_rank[x]).var(axis=0)
        plt.subplot(3, 1, 1)
        plt.plot(total_steps, feature_rank_mean, color='blue', label=f'{x} feature rank')
        plt.fill_between(total_steps, feature_rank_mean - feature_rank_std, feature_rank_mean + feature_rank_std, alpha=0.2, color='blue', label='lable1')
        plt.xlabel('Total Steps')
        plt.ylabel('Feature rank')
        plt.title(f'{x}_{name_map[idx]} : Feature rank')
        feature_rank_mean = np.array(data_weight_norm[x]).mean(axis=0)
        feature_rank_std = np.array(data_weight_norm[x]).var(axis=0)
        plt.subplot(3, 1, 2)
        plt.plot(total_steps, feature_rank_mean, color='r', label=f'{x} feature rank')
        plt.fill_between(total_steps, feature_rank_mean - feature_rank_std, feature_rank_mean + feature_rank_std, alpha=0.2, color='r', label='label2')
        plt.xlabel('Total Steps')
        plt.ylabel('weight norm')
        plt.title(f'{x}_{name_map[idx]} : weight norm')
        feature_rank_mean = np.array(data_fau[x]).mean(axis=0)
        feature_rank_std = np.array(data_fau[x]).var(axis=0)
        plt.subplot(3, 1, 3)
        plt.plot(total_steps, feature_rank_mean, color='g', label=f'{x} feature rank')
        plt.fill_between(total_steps, feature_rank_mean - feature_rank_std, feature_rank_mean + feature_rank_std, alpha=0.2, color='g', label='label3')
        plt.xlabel('Total Steps')
        plt.ylabel('FAU')
        plt.title(f'{x}_{name_map[idx]} : FAU')
        plt.tight_layout()
        plt.savefig(f"{save_folder_path}/{x}_{idx_map[idx]}")
        plt.close() 
        #plt.close() 

            #print(algo, env, seed)
            #print(len(file_path))
            # try:
            #     total_steps, v_values = read_log_file(file_path)

            #     plt.figure(figsize=(10, 6))

            #     plt.subplot(3, 1, 1)
            #     plt.plot(total_steps, [q2[0] for q2 in v_values], color='blue')
            #     plt.xlabel('Total Steps')
            #     plt.ylabel('Feature rank')
            #     plt.title('Feature rank of gaussian_policy over Total Steps')

            #     plt.subplot(3, 1, 2)
            #     plt.plot(total_steps, [q2[1] for q2 in v_values], color='red')
            #     plt.xlabel('Total Steps')
            #     plt.ylabel('weight norm')
            #     plt.title('weight norm of gaussian_policy over Total Steps')

            #     plt.subplot(3, 1, 3)
            #     plt.plot(total_steps, [q2[2] for q2 in v_values], color='green')
            #     plt.xlabel('Total Steps')
            #     plt.ylabel('FAU')
            #     plt.title('FAU of gaussian_policy over Total Steps')

            #     plt.tight_layout()

            #     save_name = os.path.splitext(file_name)[0] + '_gaussian_policy.png'
            #     save_path = os.path.join(save_folder_path, save_name)

            #     plt.savefig(save_path)
            #     plt.close() 
            # except Exception as e:
            #     print(f"Error processing file: {file_name}. Error message: {str(e)}")