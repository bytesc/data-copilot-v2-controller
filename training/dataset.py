import torch
from torch.utils.data import Dataset, DataLoader, random_split

from training.process_data import process_output_list


class DatasetSuccessRate(Dataset):
    def __init__(self, data_dict):
        self.keys = list(data_dict.keys())
        self.values = [value for value in data_dict.values()]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        x = self.keys[idx]
        y = self.values[idx][1]
        return x, y


output_file_name = "../output_store/data_log/ask_graph_1.csv"
data_list = process_output_list(output_file_name)
dataset_success_rate = DatasetSuccessRate(data_list)

# 创建DataLoader
dataloader_success_rate = DataLoader(dataset_success_rate, batch_size=1, shuffle=True)


# 使用 process_output_list 函数处理数据
output_file_name = "../output_store/data_log/ask_graph_1.csv"
data_list = process_output_list(output_file_name)
dataset_success_rate = DatasetSuccessRate(data_list)

# 划分数据集
train_size_success_rate = int(0.8 * len(dataset_success_rate))
val_size_success_rate = int(0.1 * len(dataset_success_rate))
test_size_success_rate = len(dataset_success_rate) - train_size_success_rate - val_size_success_rate

train_dataset_success_rate, val_dataset_success_rate, test_dataset_success_rate = random_split(
    dataset_success_rate, [train_size_success_rate, val_size_success_rate, test_size_success_rate])


# 创建DataLoader
train_dataloader_success_rate = DataLoader(train_dataset_success_rate, batch_size=1, shuffle=True)
val_dataloader_success_rate = DataLoader(val_dataset_success_rate, batch_size=1, shuffle=False)
test_dataloader_success_rate = DataLoader(test_dataset_success_rate, batch_size=1, shuffle=False)


# 使用DataLoader
for batch in dataloader_success_rate:
    x, y = batch
    print(f'X: {x}, Y: {y}')
    # X: ('Chart the distribution of the population of cities within the country AFG using a line graph.',), Y: tensor([0.3333], dtype=torch.float64)
