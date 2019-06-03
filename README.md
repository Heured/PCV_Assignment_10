# PCV_Assignment_10
LeNet about
## LeNet
  LeNet是卷积神经网络的祖师爷LeCun在1998年提出，用于解决手写数字识别的视觉任务。自那时起，CNN的最基本的架构就定下来了：卷积层、池化层、全连接层。如今各大深度学习框架中所使用的LeNet都是简化改进过的LeNet-5（-5表示具有5个层），和原始的LeNet有些许不同，比如把激活函数改为了现在很常用的ReLu。  
  ![emmmm](https://github.com/Heured/PCV_Assignment_10/blob/master/ImgToShow/Lenet架构.png)
  
  LeNet-5跟现有的conv->pool->ReLU的套路不同，它使用的方式是conv1->pool->conv2->pool2再接全连接层，但是不变的是，卷积层后紧接池化层的模式依旧不变。  
  ![emmmm](https://github.com/Heured/PCV_Assignment_10/blob/master/ImgToShow/Lenet流程.png)  
  
  以上图为例，对经典的LeNet-5做深入分析：  
  1.首先输入图像是单通道的28x28大小的图像，用矩阵表示就是[1,28,28]  
  
  2.第一个卷积层conv1所用的卷积核尺寸为5x5，滑动步长为1，卷积核数目为20，那么经过该层后图像尺寸变为24，28-5+1=24，输出矩阵为[20,24,24]。  
  
  3.第一个池化层pool核尺寸为2x2，步长2，这是没有重叠的max pooling，池化操作后，图像尺寸减半，变为12×12，输出矩阵为[20,12,12]。  
  
  4.第二个卷积层conv2的卷积核尺寸为5x5，步长1，卷积核数目为50，卷积后图像尺寸变为8,这是因为12-5+1=8，输出矩阵为[50,8,8]。  
  
  5.第二个池化层pool2核尺寸为2x2，步长2，这是没有重叠的max pooling，池化操作后，图像尺寸减半，变为4×4，输出矩阵为[50,4,4]。  
  
  6.pool2后面接全连接层fc1，神经元数目为500，再接relu激活函数。  
  
  7.再接fc2，神经元个数为10，得到10维的特征向量，用于10个数字的分类训练，送入softmaxt分类，得到分类结果的概率output。  
  
  

### 所用数据
  本次实验使用数据为Mnist数据集，包括60000张图像的训练集以及10000张图像的测试集。每张图片大小为 28x28 像素，图片中纯黑色的像素值为0，纯白色像素值为1。数据集的标签是长度为10的一维数组，数组中每个元素索引号表示对应数字出现的概率。  
  ![emmmm](https://github.com/Heured/PCV_Assignment_10/blob/master/ImgToShow/44444.PNG)  
  
  如上图，每个种有多种写法图像，这是为了提高网络模型的鲁棒性。
  
 
### 过程

![emmmm](https://github.com/Heured/PCV_Assignment_10/blob/master/ImgToShow/训练过程.PNG)  
  
 

