import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Reshape, LeakyReLU, Input, BatchNormalization,Dropout
from keras.optimizers import Adam
from keras.regularizers import L1L2
import matplotlib.pyplot as plt


class GAN:
    def __init__(self):
        # shape of input image
        self.image_shape = (28, 28, 1)
        self.noise_shape = (20,)
        self.optimiser = Adam(0.0002, 0.5)

        # now we create both our networks

        self.generator = self.build_generator()
        self.generator.compile(optimizer=self.optimiser, loss='binary_crossentropy')

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer=self.optimiser, loss='binary_crossentropy', metrics=['accuracy'])

        noise = Input(self.noise_shape)
        img = self.generator(noise)
        self.discriminator.trainable = False

        validity = self.discriminator(img)

        self.combined = Model(noise, validity)    # 讓假資料流進 gen
        self.combined.compile(optimizer=self.optimiser, loss='binary_crossentropy')

    def build_generator(self):
        # the generator will be a fully connected network
        # with input noise vector of shape (20,)
        # the output of the generator will be a 28x28 image
       
        model = Sequential()
        model.add(Dense(256, input_shape=self.noise_shape))
        model.add(LeakyReLU())
        model.add(BatchNormalization(momentum=0.8))
       # model.add(Dropout(0.50))

        model.add(Dense(512))
        model.add(LeakyReLU())
        model.add(BatchNormalization(momentum=0.8))
        #model.add(Dropout(0.50))

        model.add(Dense(784, activation='tanh'))
        model.add(Reshape(self.image_shape))
        print('Generator model : ')
        print(model.summary())

        noise = Input(self.noise_shape)
        gen_img = model(noise)

        return Model(noise, gen_img)

    def build_discriminator(self):
        # this is a simple FCN
        # input for the discriminator is a 28x28 image

        model = Sequential()
        model.add(Flatten(input_shape=self.image_shape))        # 28 x 28 變成一維

        model.add(Dense(512))
        model.add(LeakyReLU())
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dropout(0.25))

        model.add(Dense(256))
        model.add(LeakyReLU())
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dropout(0.25))

        model.add(Dense(1, activation='sigmoid'))       # 輸出 1
        print('Discriminator model : ')
        print(model.summary())

        img = Input(self.image_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, X, epochs=1, batch_size=64):

        
        X_train = (X.astype(np.float32)-127.5) / 127.5
        X_train = X_train.reshape((X.shape[0], 28, 28, 1))
        
       

        for epoch in range(epochs + 1):
            # -----------train discriminator ------------
           
            idx = np.random.randint(0, X.shape[0], batch_size)   # 隨機抓 64 筆，idx 是一個nd.array
            real_imgs = X_train[idx]
            noise = np.random.normal(0, 1, (batch_size, 20))     # { 20 } x 64
            fakes = self.generator.predict(noise)                # 從 noise 裡面生成資料

                    ####

                            #################       real_imgs 
                            #    Dataset    #   ------------------|
                            #################                     |               #################
                    #                                             |——————————>    #      DIS      #
        #      noise        #################          fakes      |               #################
        # ——————————————    #      GEN      #   ------------------|
                            #################

                    ####

            d_loss_real = self.discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))  # 先用真是資料 train dis， 再假資料過一遍
            d_loss_fake = self.discriminator.train_on_batch(fakes, np.zeros((batch_size, 1)))    # data match label
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)      # 看大家的過兩次分別的 loss
            
            # print(d_loss_real)
            # print("============")
            # print(d_loss_fake)
            # print(np.zeros((64, 1)))

            # --------------train generator -------------------

            # Train the generator (to have the discriminator label samples as valid)
            # try to fool the discriminator
            noise = np.random.normal(0, 1, (batch_size, 20))
            valid_y = [1] * batch_size
            g_loss = self.combined.train_on_batch(noise, valid_y)
            # Hung-Yi Li : 把 gen 和 dis 合起來看作是一個巨大的 network，input 是 vector，就是 noise
            # 因此理解上是 input 了一張由 gen 產生的圖片，實際上是 一個 vector。 combined 就是合併兩個。
            
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            if epoch % 1000 == 0:
                self.save_imgs(epoch)


    def save_imgs(self, epoch):
        r, c = 3, 3
        noise = np.random.normal(0, 1, (r * c, 20))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap=None)    
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("mona_lisa%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    mona_lisa = np.load("mona_lisa.npy")
    gan.train(X=mona_lisa)
