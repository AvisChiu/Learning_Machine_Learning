BasicGAN
--


**標準的用 Keras 搭建 GAN 網絡**    
**目的: 為了熟悉一下怎麼使用 Keras, 和 GAN 的一些基本認識**    
**Dataset**
https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap?pli=1

<div align=center> <img src="https://github.com/AvisChiu/Machine_learning_practice/blob/master/BasicGAN/figure/GAN1.png" width="500",height="500"/></div>
<br/>

<div align=center> <img src="https://github.com/AvisChiu/Machine_learning_practice/blob/master/BasicGAN/figure/GAN2.png" width="500",height="500"/></div>

<br/>


**GAN 的實際訓練流程**

  * 1. 從 真實資料 裡面 sample 一些資料出來。
  * 2. 從 noise 裡面 sample 一些資料出來。
```
idx = np.random.randint(0, X.shape[0], batch_size)   # 隨機抓 64 筆，idx 是一個 nd.array
real_imgs = X_train[idx]
noise = np.random.normal(0, 1, (batch_size, 20))     # { 20 } x 64
fakes = self.generator.predict(noise)                # 從 noise 裡面生成資料
```
  * 3. 訓練 Discriminator
    * discriminator 的訓練方法如上圖，真實資料 label 為 1， 假資料 label 為 0
```
d_loss_real = self.discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))  # 先用真是資料 train dis， 再假資料過一遍
d_loss_fake = self.discriminator.train_on_batch(fakes, np.zeros((batch_size, 1)))    # data match label
d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)      # 看大家的過兩次分別的 loss
```
  * 4. 訓練 Generator
    * 這邊要參考 NTU 李宏毅教授的講法，generator 的訓練是吧 gen-dis 合併起來看作是一個網絡，於是 input 就是一個 vector ， label 為 1
```
noise = np.random.normal(0, 1, (batch_size, 20))
valid_y = [1] * batch_size
g_loss = self.combined.train_on_batch(noise, valid_y)
```

<br/>


**Generator 網絡結構**

  * 20 --> 256 --> 512 --> 784 
  * 因為真實資料的大小是 28 x 28，因此 generator 出來的要與之對應， 最後 reshape

```
model.add(Dense(256, input_shape=self.noise_shape))   #  noise_shape = (20,)
model.add(LeakyReLU())
model.add(BatchNormalization(momentum=0.8))

model.add(Dense(512))
model.add(LeakyReLU())
model.add(BatchNormalization(momentum=0.8))

model.add(Dense(784, activation='tanh'))
model.add(Reshape(self.image_shape))
        
```

**Discriminator 網絡結構** 

  * 注意 Discriminator 最後輸出的是一個 “分數”
```
model = Sequential()
model.add(Flatten(input_shape=self.image_shape))    # 28 x 28 變成一維（展平）
model.add(Dense(512))
model.add(LeakyReLU())
       
model.add(Dense(256))
model.add(LeakyReLU())

model.add(Dense(1, activation='sigmoid'))           # 輸出 1
     
```

**輸入 Input**

```
self.image_shape = (28, 28, 1)   
self.noise_shape = (20,)
```

**實驗結果（沒 1000 個 epoch 輸出一次）**

<div align=center> <img src="https://github.com/AvisChiu/Machine_learning_practice/blob/master/BasicGAN/figure/mnist_0.png" width="600",height="600"/></div>
<br/>

<div align=center> <img src="https://github.com/AvisChiu/Machine_learning_practice/blob/master/BasicGAN/figure/mona_lisa_1000.png" width="600",height="600"/></div>
<br/>

<div align=center> <img src="https://github.com/AvisChiu/Machine_learning_practice/blob/master/BasicGAN/figure/mona_lisa_2000.png" width="600",height="600"/></div>
<br/>


<div align=center> <img src="https://github.com/AvisChiu/Machine_learning_practice/blob/master/BasicGAN/figure/mona_lisa_3000.png" width="600",height="600"/></div>
<br/>

<div align=center> <img src="https://github.com/AvisChiu/Machine_learning_practice/blob/master/BasicGAN/figure/mona_lisa_4000.png" width="600",height="600"/></div>
<br/>


<div align=center> <img src="https://github.com/AvisChiu/Machine_learning_practice/blob/master/BasicGAN/figure/mona_lisa_5000.png" width="600",height="600"/></div>
<br/>

<div align=center> <img src="https://github.com/AvisChiu/Machine_learning_practice/blob/master/BasicGAN/figure/mona_lisa_6000.png" width="600",height="600"/></div>
<br/>

