import tensorflow.python.keras.backend

import autoencoder
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
import tensorflow.keras as tfk
import numpy as np


class GAN:
    def __init__(self):
        self.kernel_size = (3, 3)
        self.generator = self.createGenerator()
        self.discriminator = self.createDiscriminator()
        self.palette_shape = (10, 1)  # only hue channel
        self.gray_picture_shape = (256, 256, 1)

    def createGenerator(self):
        # return autoencoder.Autoencoder()

        input_gray_pic = layers.Input(shape=(256, 256, 1))
        input_palette = layers.Input(shape=(256, 256, 10))

        # Hinput = layers.Lambda(self.normalizeHSV)(input_color_hue)
        # model = layers.BatchNormalization()(input_gray_pic)
        # model = layers.Conv2D(filters=10, kernel_size=self.kernel_size, padding='same', activation='relu')(input_gray_pic) # 256x256x8
        # model_pal = layers.Lambda(self.myNormPalette(256))
        # concat = layers.concatenate([model, model_pal], axis=2)
        # concat = layers.Conv2D(filters=10, kernel_size=self.kernel_size, padding='same', activation='relu')(concat)

        model = layers.Conv2D(filters=10, kernel_size=self.kernel_size, padding='same', activation='relu')(
            input_gray_pic)  # 256x256x8
        model = layers.concatenate([model, input_palette], axis=3)
        model = layers.Conv2D(filters=10, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.BatchNormalization()(model)

        model = layers.Conv2D(filters=32, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=32, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        # model = layers.MaxPooling2D(pool_size=(2, 2))(model)
        model_pal = layers.MaxPooling2D(pool_size=(1, 1))(input_palette)

        model = layers.Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=32, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        model = layers.Conv2D(filters=10, kernel_size=self.kernel_size, padding='same', activation='relu')(
            model)  # 256x256x8
        model = layers.concatenate([model, model_pal], axis=3)
        model = layers.Conv2D(filters=10, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.BatchNormalization()(model)

        model = layers.Conv2D(filters=32, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        # model = layers.MaxPooling2D(pool_size=(2, 2))(model)
        # model_pal = layers.MaxPooling2D(pool_size=(2, 2))(model_pal)

        model = layers.Conv2D(filters=128, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=32, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        model = layers.Conv2D(filters=10, kernel_size=self.kernel_size, padding='same', activation='relu')(
            model)  # 256x256x8
        model = layers.concatenate([model, model_pal], axis=3)
        model = layers.Conv2D(filters=10, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.BatchNormalization()(model)

        model = layers.Conv2D(filters=32, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        # model = layers.Conv2DTranspose(filters=32, kernel_size=self.kernel_size, strides=2, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=16, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=32, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        model = layers.Conv2D(filters=3, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        # model = layers.Conv2DTranspose(filters=3, kernel_size=self.kernel_size, strides=2, padding='same',
        #                               activation='relu')(model)
        model = layers.Conv2D(filters=3, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=3, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=3, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        model = Model(inputs=[input_gray_pic, input_palette], outputs=[model])
        """
        # CNN LAYERS
        model = layers.Conv2D(filters=8, kernel_size=self.kernel_size, padding='same', activation='relu')(
            input_gray_pic)

        # model = layers.MaxPooling2D(pool_size=(2, 2))(model)

        model = layers.Conv2D(filters=16, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        # model = layers.MaxPooling2D(pool_size=(2, 2))(model)

        model = layers.Conv2D(filters=24, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        model = layers.Conv2D(filters=32, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        # model = layers.MaxPooling2D(pool_size=(2, 2))(model)

        model = layers.Conv2D(filters=48, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        model = layers.Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        # model = layers.MaxPooling2D(pool_size=(2, 2))(model)

        model = layers.Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        model = layers.Conv2D(filters=32, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        # model = layers.MaxPooling2D(pool_size=(2, 2))(model)

        model = layers.Conv2D(filters=16, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        # ------------------------------------------------------------------------------------------------------
        model = layers.Conv2D(filters=128, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        # model = layers.Conv2DTranspose(filters=8, kernel_size=self.kernel_size, strides=2, padding='same', activation='relu')(model)

        model = layers.Conv2D(filters=128, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        # model = layers.Conv2DTranspose(filters=16, kernel_size=self.kernel_size, strides=2, padding='same', activation='relu')(model)

        model = layers.Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        # model = layers.Conv2DTranspose(filters=32, kernel_size=self.kernel_size, strides=2, padding='same', activation='relu')(model)

        model = layers.Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=32, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=16, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=8, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=3, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        model = layers.Lambda(lambda x: x * 255)(model)


"""

        model.summary()

        plot_model(model, to_file='networkStructure/generator.png', show_shapes=True, show_layer_names=True)
        return model

    def createDiscriminator(self):
        model = Sequential()
        model.add(layers.Conv2D(16, self.kernel_size, strides=(2, 2), padding='same', input_shape=[256, 256, 3]))
        model.add(layers.Conv2D(32, self.kernel_size, strides=(2, 2), padding='same'))
        model.add(layers.Conv2D(64, self.kernel_size, strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, self.kernel_size, padding='same'))
        model.add(layers.Conv2D(32, self.kernel_size, padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(32))
        model.add(layers.Dense(16))
        model.add(layers.Dense(1))

        model.summary()

        plot_model(model, to_file='networkStructure/discriminator.png', show_shapes=True, show_layer_names=True)

        return model


if __name__ == '__main__':
    gan = GAN()
