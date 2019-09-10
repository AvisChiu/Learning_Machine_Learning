BasicGAN
--
**標準的用 Keras 搭建 GAN 網絡**
**目的: 為了熟悉一下怎麼使用 Keras, 和 GAN 的一些基本認識**

<div align=center> <img src="https://github.com/AvisChiu/Machine_learning_practice/blob/master/BasicGAN/figure/GAN1.png" width="800",height="800"/></div>
<br/>

<div align=center> <img src="https://github.com/AvisChiu/Machine_learning_practice/blob/master/BasicGAN/figure/GAN2.png
" width="800",height="800"/></div>
<br/>

* **GAN 的實際訓練流程

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
