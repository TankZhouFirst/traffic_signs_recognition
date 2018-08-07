## 说明
第一次上传代码。本代码使用 TensorFlow 构建了一个交通信号识别的神经网络。[数据集下载](https://btsd.ethz.ch/shareddata/)。
下载 BelgiumTS for Classification (cropped images) 右边的两个压缩包： BelgiumTSC_Training 和 BelgiumTSC_Testing。
数据集中，每个类别用一串数字表示，共 62 类，但是每类别下图片数可能不等。另外，你会发现这些文件的文件扩展名是 .ppm，即 Portable Pixmap Format。

## 训练
直接运行 train.py 即可。

## 问题
超参数选择并不很合适，也没有仔细调试，仅用于熟悉 CNN。

## 结果
![image](https://github.com/TankZhouFirst/traffic_signs_recognition/blob/master/result.png)
