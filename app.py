import torch
import torch.nn as nn
import pandas as pd
import re

# Đọc file CSV
df = pd.read_csv("./data/SMS_train.csv", encoding='ANSI')

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

# Áp dụng hàm convert_to_boolean cho từng dòng trong cột 'tokenized_text'
df['tensor'] = df['tokenized_text'].apply(lambda x: convert_to_boolean(x))

print(df.head())
