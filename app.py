import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import re
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split

# Đọc file CSV
df = pd.read_csv("./data/SMS.csv", encoding='ANSI')

# Tokenize từng dòng trong cột 'Message_body'
df['tokenized_text'] = df['Message_body'].apply(lambda x: re.findall(r'\b\w+\b', x.lower()))

# Kết hợp tất cả các danh sách từ thành một danh sách duy nhất
all_words = df['tokenized_text'].sum()

# Loại bỏ các từ trùng lặp và tạo một danh sách từ duy nhất
list_word = list(set(all_words))

# Đặt kích thước vector
len_vector = len(list_word)

# Tạo từ điển để tra cứu chỉ số từ
word_to_index = {word: index for index, word in enumerate(list_word)}

def convert_to_boolean(arr):
    # Tạo tensor chứa toàn số 0 với kích thước len_vector
    ts = torch.zeros(len_vector)
    # Cập nhật tensor với giá trị 1 ở các vị trí từ có mặt
    for word in arr:
        if word in word_to_index:
            ts[word_to_index[word]] = 1
    return ts

# Chuyển đổi tin nhắn thành tensor boolean
df['tensor'] = df['tokenized_text'].apply(lambda x: convert_to_boolean(x))

# Mã hóa nhãn
df['label_encoded'] = df['Label'].apply(lambda x: 1 if x == 'Spam' else 0)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra bằng PyTorch
X = torch.stack(df['tensor'].tolist())
y = torch.tensor(df['label_encoded'].tolist(), dtype=torch.float32)

dataset = torch.utils.data.TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Load data using DataLoader for batching (optional)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


class BinaryClassificationModel(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassificationModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        return self.sigmoid(x)

model = BinaryClassificationModel(input_size=len_vector)

# Define the loss and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 3000
for epoch in range(epochs):
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        for i in range(len(inputs)):
            outputs = model(inputs[i]).squeeze()
            if labels.ndim == 0:  # Handle the case where labels are scalars
              labels[i] = labels[i].unsqueeze(0)
            loss = criterion(outputs, labels[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        break
