import datetime

import torch
from torch import nn
from torch.optim import Adam
from transformers import BertTokenizer, BertModel

from training.dataset import dataloader_success_rate, train_dataloader_success_rate
from training.dataset import test_dataloader_success_rate, val_dataloader_success_rate

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
for epoch in range(10):  # 训练10个epoch
    train_loss = 0.0
    for batch in train_dataloader_success_rate:
        x, y = batch
        y = y.float().to(device)  # Convert y to float and then move to device

        # 编码文本数据
        inputs = tokenizer(x[0], return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # 前向传播
        outputs = model(input_ids, attention_mask)
        loss = loss_function(outputs.float(), y.unsqueeze(1))  # Cast outputs to float

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 验证模型
    val_loss = 0.0
    model.eval()
    total_abs_error = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in test_dataloader_success_rate:
            x, y = batch
            y = y.to(device)

            inputs = tokenizer(x[0], return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)
            predicted = outputs.flatten()

            # 计算预测值和实际值之间的绝对误差
            abs_error = torch.abs(predicted - y)
            total_abs_error += abs_error.sum().item()
            total_samples += len(y)

    # 计算平均绝对误差
    mean_abs_error = total_abs_error / total_samples
    print(f'Epoch: {epoch}, Train Loss: {train_loss / len(train_dataloader_success_rate)},'
          f' Val Loss: {val_loss / len(val_dataloader_success_rate)},'
          f' Mean Absolute Error: {mean_abs_error}')

# 测试模型
model.eval()
with torch.no_grad():
    for batch in test_dataloader_success_rate:
        x, y = batch
        y = y.to(device)

        inputs = tokenizer(x[0], return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask)
        print(f'Predicted: {outputs}, Actual: {y}')


timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
torch.save(model.state_dict(), f'model_{timestamp}.pth')