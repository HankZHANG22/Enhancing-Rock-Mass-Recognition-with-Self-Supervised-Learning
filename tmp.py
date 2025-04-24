import os


fanhua = "/home/langchao16/data/准泛化测试样本2024"
fanhuas = []
for folder in os.listdir(fanhua):
    folder_path = os.path.join(fanhua, folder)
    for folder2 in os.listdir(folder_path):
        fanhuas.append(folder2)

train_set = os.listdir('/home/langchao16/data/train_set_detected')
for fanhua in fanhuas:
    if fanhua in train_set:
        print(fanhua)
