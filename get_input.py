import os
from random import shuffle
import numpy as np
from skimage import io, transform
from skimage.color import rgb2gray

def get_data(data_dir):
    dir_list = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    images = []
    labels = []
    # 开始迭代所有路径
    for d in dir_list:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.ppm')]
        # 开始迭代文件
        for f in file_names:
            images.append(io.imread(f))
            labels.append(int(d))
    return images, labels

# 调整图像大小
def reshape_img(images_list):
    return [transform.resize(image, (28, 28)) for image in images_list]

# 将图像调整为灰度
def gray_img(img_list):
    img_list = np.array(img_list)
    return rgb2gray(img_list)

# 获取数据
def get_input(data_dir):
    images, labels = get_data(data_dir)
    images = reshape_img(images)
    images = gray_img(images)

    index = list(range(len(labels)))
    shuffle(index)
    shuffled_images = []
    shuffled_labels = []

    for i in index:
        shuffled_images.append(images[i])
        tmp_label = [0 for i in range(62)]
        tmp_label[labels[i]] = 1
        shuffled_labels.append(tmp_label)

    return shuffled_images, shuffled_labels