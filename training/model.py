import datetime

import torch
from torch import nn
from torch.optim import Adam
from transformers import BertTokenizer, BertModel


from utils.write_csv import write_csv_from_list

# 初始化BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('D:/IDLE/projects/models/bert-base-multilingual-uncased/')
bert_model = BertModel.from_pretrained('D:/IDLE/projects/models/bert-base-multilingual-uncased/')




class BertRegressionModel(nn.Module):
    def __init__(self):
        super(BertRegressionModel, self).__init__()
        self.bert = bert_model
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.regressor(pooled_output)


def training():
    from training.dataset import dataloader_success_rate, train_dataloader_success_rate
    from training.dataset import test_dataloader_success_rate, val_dataloader_success_rate
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_file_name = f'./train_logs/log_{timestamp}.csv'

    # 实例化模型
    model = BertRegressionModel()

    # 将模型移动到相应的设备上（CPU或GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义损失函数和优化器
    loss_function = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-5)

    # 训练模型
    model.train()
    threshold = 0.1
    for epoch in range(10000):
        train_loss = 0.0
        correct_predictions_train = 0
        total_samples_train = 0
        sum_abs_error_train = 0.0  # 累加绝对误差

        for batch in train_dataloader_success_rate:
            x, y = batch
            y = y.float().to(device)  # Convert y to float and then move to device
            total_samples_train += len(y)

            inputs = tokenizer(x[0], return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_function(outputs.float(), y.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            abs_error = torch.abs(outputs.flatten() - y)
            correct_predictions_train += (abs_error < threshold).sum().item()
            sum_abs_error_train += abs_error.sum().item()  # 累加绝对误差

        train_accuracy = correct_predictions_train / total_samples_train
        train_avg_abs_error = sum_abs_error_train / total_samples_train
        f'Epoch: {epoch}, Train Loss: {train_loss / len(train_dataloader_success_rate)}, '
        f' Train Accuracy: {train_accuracy}, Train Average Error: {train_avg_abs_error}, '

        model.eval()
        correct_predictions_val = 0
        sum_abs_error_val = 0.0  # 累加绝对误差
        total_samples_val = 0

        with torch.no_grad():
            for batch in val_dataloader_success_rate:
                x, y = batch
                y = y.to(device)
                total_samples_val += len(y)

                inputs = tokenizer(x[0], return_tensors="pt", padding=True, truncation=True, max_length=512)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)

                outputs = model(input_ids, attention_mask)
                predicted = outputs.flatten()

                abs_error = torch.abs(predicted - y)
                correct_predictions_val += (abs_error < threshold).sum().item()
                sum_abs_error_val += abs_error.sum().item()  # 累加绝对误差

        val_accuracy = correct_predictions_val / total_samples_val
        val_avg_abs_error = sum_abs_error_val / total_samples_val

        print(
            f' Validation Accuracy: {val_accuracy}, Validation Average Error: {val_avg_abs_error}')

        write_csv_from_list(log_file_name, [epoch, train_loss / len(train_dataloader_success_rate),
                                            train_accuracy, train_avg_abs_error,
                                            val_accuracy, val_avg_abs_error])

        if epoch % 10 == 9:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            torch.save(model.state_dict(), f'./saves/model_{epoch}_{timestamp}.pth')

    # 测试模型
    model.eval()
    correct_predictions = 0
    sum_abs_error_test = 0.0  # 累加绝对误差
    total_samples_test = 0

    with torch.no_grad():
        for batch in test_dataloader_success_rate:
            x, y = batch
            y = y.to(device)
            total_samples_test += len(y)

            inputs = tokenizer(x[0], return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)
            predicted = outputs.flatten()

            abs_error = torch.abs(predicted - y)
            correct_predictions += (abs_error < threshold).sum().item()
            sum_abs_error_test += abs_error.sum().item()  # 累加绝对误差

            print(f'Predicted: {outputs}, Actual: {y}')

        test_accuracy = correct_predictions / total_samples_test
        test_avg_abs_error = sum_abs_error_test / total_samples_test  # 计算平均绝对误差
        print(f'Test Accuracy: {test_accuracy}, Test Average Absolute Error: {test_accuracy}')
        write_csv_from_list(log_file_name, [test_accuracy, test_accuracy])

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    torch.save(model.state_dict(), f'./saves/model_final_{timestamp}.pth')


if __name__ == "__main__":
    training()
