# PCV_Assignment_10
LeNet about
## LeNet
  LeNet是卷积神经网络的祖师爷LeCun在1998年提出，用于解决手写数字识别的视觉任务。自那时起，CNN的最基本的架构就定下来了：卷积层、池化层、全连接层。如今各大深度学习框架中所使用的LeNet都是简化改进过的LeNet-5（-5表示具有5个层），和原始的LeNet有些许不同，比如把激活函数改为了现在很常用的ReLu。  
  ![emmmm](https://github.com/Heured/PCV_Assignment_10/blob/master/ImgToShow/Lenet架构.png)
  
  LeNet-5跟现有的conv->pool->ReLU的套路不同，它使用的方式是conv1->pool->conv2->pool2再接全连接层，但是不变的是，卷积层后紧接池化层的模式依旧不变。  
  ![emmmm](https://github.com/Heured/PCV_Assignment_10/blob/master/ImgToShow/Lenet流程.png)  
  
  以上图为例，对经典的LeNet-5做深入分析：  
  1.首先输入图像是单通道的28x28大小的图像，用矩阵表示就是[1,28,28]  
  
  ```python
  
  # 获取训练集
  def getTrain():
    train=[[],[]] # 指定训练集的格式，一维为输入数据，一维为其标签
    # 读取所有训练图像，作为训练集
    train_root="mnist_train" 
    labels = os.listdir(train_root)
    for label in labels:
        imgpaths = os.listdir(os.path.join(train_root,label))
        for imgname in imgpaths:
            img = cv2.imread(os.path.join(train_root,label,imgname),0)
            array = np.array(img).flatten() # 将二维图像平铺为一维图像
            array=MaxMinNormalization(array)
            train[0].append(array)
            label_ = [0,0,0,0,0,0,0,0,0,0]
            label_[int(label)] = 1
            train[1].append(label_)
    train = shuff(train)
    return train
    
# 获取测试集
def getTest():
    test=[[],[]] # 指定训练集的格式，一维为输入数据，一维为其标签
    # 读取所有训练图像，作为训练集
    test_root="mnist_test" 
    labels = os.listdir(test_root)
    for label in labels:
        imgpaths = os.listdir(os.path.join(test_root,label))
        for imgname in imgpaths:
            img = cv2.imread(os.path.join(test_root,label,imgname),0)
            array = np.array(img).flatten() # 将二维图像平铺为一维图像
            array=MaxMinNormalization(array)
            test[0].append(array)
            label_ = [0,0,0,0,0,0,0,0,0,0]
            label_[int(label)] = 1
            test[1].append(label_)
    test = shuff(test)
    return test[0],test[1]
    
  # 打乱数据集
  def shuff(data):
    temp=[]
    for i in range(len(data[0])):
        temp.append([data[0][i],data[1][i]])
    import random
    random.shuffle(temp)
    data=[[],[]]
    for tt in temp:
        data[0].append(tt[0])
        data[1].append(tt[1])
    return data

count = 0
  ```
  
  2.第一个卷积层conv1所用的卷积核尺寸为5x5，滑动步长为1，卷积核数目为20，那么经过该层后图像尺寸变为24，28-5+1=24，输出矩阵为[20,24,24]。  
  
  ```python
w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32]) # 每个输出通道都有一个对应的偏置量
# 我们把x变成一个4d 向量其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(灰度图的通道数为1，如果是RGB彩色图，则为3)
x_image = tf.reshape(x,[-1,28,28,1])
# 因为只有一个颜色通道，故最终尺寸为[-1，28，28，1]，前面的-1代表样本数量不固定，最后的1代表颜色通道数量
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1) # 使用conv2d函数进行卷积操作，非线性处理
  ```
  
  
  3.第一个池化层pool核尺寸为2x2，步长2，这是没有重叠的max pooling，池化操作后，图像尺寸减半，变为12×12，输出矩阵为[20,12,12]。  
  
  ```python
h_pool1 = max_pool_2x2(h_conv1)                          # 对卷积的输出结果进行池化操作
  ```
  
  4.第二个卷积层conv2的卷积核尺寸为5x5，步长1，卷积核数目为50，卷积后图像尺寸变为8,这是因为12-5+1=8，输出矩阵为[50,8,8]。  
  
  ```python
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)# 输入的是第一层池化的结果
  ```
  
  5.第二个池化层pool2核尺寸为2x2，步长2，这是没有重叠的max pooling，池化操作后，图像尺寸减半，变为4×4，输出矩阵为[50,4,4]。  
  
  ```python
h_pool2 = max_pool_2x2(h_conv2)
  ```
  
  6.pool2后面接全连接层fc1，神经元数目为500，再接relu激活函数。  
  
  ```python
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1,7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# 使用dropout，防止过度拟合
keep_prob = tf.placeholder(tf.float32, name="keep_prob")# placeholder是占位符
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  ```
  
  7.再接fc2，神经元个数为10，得到10维的特征向量，用于10个数字的分类训练，送入softmaxt分类，得到分类结果的概率output。  
  
  ```python
w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2, name="y-pred")
  ```
  
  

### 所用数据
  本次实验使用数据为Mnist数据集，包括60000张图像的训练集以及10000张图像的测试集。每张图片大小为 28x28 像素，图片中纯黑色的像素值为0，纯白色像素值为1。数据集的标签是长度为10的一维数组，数组中每个元素索引号表示对应数字出现的概率。  
  ![emmmm](https://github.com/Heured/PCV_Assignment_10/blob/master/ImgToShow/44444.PNG)  
  
  如上图，每个种有多种写法图像，这是为了提高网络模型的鲁棒性。
  
 
### 过程

![emmmm](https://github.com/Heured/PCV_Assignment_10/blob/master/ImgToShow/训练过程.PNG)  
  
![emmmm](https://github.com/Heured/PCV_Assignment_10/blob/master/ImgToShow/曲线.PNG)  
  
从图上可看出，在训练次数到60~80时逐渐收敛。  
  
结果：  
![emmmm](https://github.com/Heured/PCV_Assignment_10/blob/master/ImgToShow/结果01.PNG)  
![emmmm](https://github.com/Heured/PCV_Assignment_10/blob/master/ImgToShow/结果02.PNG)  
![emmmm](https://github.com/Heured/PCV_Assignment_10/blob/master/ImgToShow/结果03.PNG)  
  
也许是网络的鲁棒性过高，导致特征相似的两种数字被误判。

  
 

