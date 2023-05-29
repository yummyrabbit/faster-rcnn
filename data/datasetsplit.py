# 数据集划分集类
import os
from sklearn.model_selection import train_test_split

image_path = r'E:/class/深度学习/VOC2012/JPEGImages'
image_list = os.listdir(image_path)
names = []

for i in image_list:
    names.append(i.split('.')[0])  # 获取图片名
trainval, test = train_test_split(names, test_size=0.2, shuffle=17125)  # shuffle()中是图片总数目
validation, train = train_test_split(trainval, test_size=0.75, shuffle=17125)

with open('E:/class/深度学习/VOC2012/ImageSets/Main/trainval.txt', 'w') as f:
    for i in trainval:
        f.write(i + '\n')
with open('E:/class/深度学习/VOC2012/ImageSets/Main/test.txt', 'w') as f:
    for i in test:
        f.write(i + '\n')
with open('E:/class/深度学习/VOC2012/ImageSets/Main/validation.txt', 'w') as f:
    for i in validation:
        f.write(i + '\n')
with open('E:/class/深度学习/VOC2012/ImageSets/Main/train.txt', 'w') as f:
    for i in train:
        f.write(i + '\n')
print(len(test),len(validation),len(train),len(trainval))
