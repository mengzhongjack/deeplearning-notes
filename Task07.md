整个神经网络，我觉得最重要的点有:
* 反向传播
* 求最快的梯度，向量化的操作能够节省时间和成本
* 神经网络的特征设计

## 卷积神经网络
* 卷积神经网络是个里程碑，与普通的神经网络有着很大的区别
  

* 参数共享机制
在卷积层中每个神经元连接数据窗的权重是固定的，每个神经元只关注一个特性。神经元就是图像处理中的滤波器，比如边缘检测专用的Sobel滤波器，即卷积层的每个滤波器都会有自己所关注一个图像特征，比如垂直边缘，水平边缘，颜色，纹理等等，这些所有神经元加起来就好比就是整张图像的特征提取器集合。

