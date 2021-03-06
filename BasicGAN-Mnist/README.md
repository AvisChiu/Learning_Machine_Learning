BasicGAN-Mnist
--

**標準的用 Keras 搭建 GAN 網絡**    
**目的: 繼續熟悉一下怎麼使用 Keras, 和 GAN 的一些基本認識， 進一步知道怎麼灌入資料集**    
**下一步: 可以試試直接收集的資料，需要做前處理**
**Dataset: 直接 call keras 下載**

```
from keras.datasets.mnist import load_data
```
<br/>

**資料要先經過一些處理**
```
def load_real_samples(self):
    (trainX, _), (_, _) = load_data()
    X = expand_dims(trainX, axis=-1)
    return X
```
<br/>

**資料要先經過一些處理**
其他跟 BaiscGAN 一樣，只不過是換了一個「資料集」


<br/>
**資料要先經過一些處理（擷取每 10000 個 epoch ）**
**(1000, 10000,20000,30000,40000,50000)**

<div align=center> <img src="https://github.com/AvisChiu/Machine_learning_practice/blob/master/BasicGAN-Mnist/figure/mnist_1000.png" width="600",height="600"/></div>
<br/>

<div align=center> <img src="https://github.com/AvisChiu/Machine_learning_practice/blob/master/BasicGAN-Mnist/figure/mnist_10000.png" width="600",height="600"/></div>
<br/>

<div align=center> <img src="https://github.com/AvisChiu/Machine_learning_practice/blob/master/BasicGAN-Mnist/figure/mnist_20000.png" width="600",height="600"/></div>
<br/>

<div align=center> <img src="https://github.com/AvisChiu/Machine_learning_practice/blob/master/BasicGAN-Mnist/figure/mnist_30000.png" width="600",height="600"/></div>
<br/>

<div align=center> <img src="https://github.com/AvisChiu/Machine_learning_practice/blob/master/BasicGAN-Mnist/figure/mnist_40000.png" width="600",height="600"/></div>
<br/>

<div align=center> <img src="https://github.com/AvisChiu/Machine_learning_practice/blob/master/BasicGAN-Mnist/figure/mnist_50000.png" width="600",height="600"/></div>
<br/>
