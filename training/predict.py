import torch
from transformers import BertTokenizer, BertModel

# 加载模型和分词器
tokenizer = BertTokenizer.from_pretrained(r'D:/IDLE/projects/models/bert-base-multilingual-uncased/')
bert_model = BertModel.from_pretrained(r'D:/IDLE/projects/models/bert-base-multilingual-uncased/')

from training.model import BertRegressionModel

# 初始化模型
model = BertRegressionModel()

# 加载模型权重
model_path = r"D:\IDLE\projects\copilot-data-backup\result\241024\model_59_20241025022023.pth"

# 确定设备是CPU还是GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 将模型移动到相应的设备上
model.to(device)


# 假设我们有一个字符串列表作为输入
texts = ["Your text here", "Another text here"]

# 使用模型进行预测
for text in texts:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        predicted = outputs.flatten().cpu().numpy()  # 将预测结果移至CPU并转换为numpy数组

    print(f'Predicted: {predicted}')
