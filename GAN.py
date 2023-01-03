import autoencoder
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model


class GAN:
    def __init__(self):
        self.kernel_size = (3, 3)
        self.generator = self.createGenerator()
        self.discriminator = self.createDiscriminator()

    def createGenerator(self):
        # return autoencoder.Autoencoder()

        input_gray_pic = layers.Input(shape=(256, 256, 1))

        # Hinput = layers.Lambda(self.normalizeHSV)(input_color_hue)
        model = layers.Lambda(lambda x: x / 255)(input_gray_pic)

        # CNN LAYERS
        model = layers.Conv2D(filters=8, kernel_size=self.kernel_size, padding='same', activation='relu')(
            input_gray_pic)

        model = layers.MaxPooling2D(pool_size=(2, 2))(model)

        model = layers.Conv2D(filters=16, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        model = layers.MaxPooling2D(pool_size=(2, 2))(model)

        model = layers.Conv2D(filters=24, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        model = layers.Conv2D(filters=32, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        model = layers.MaxPooling2D(pool_size=(2, 2))(model)

        model = layers.Conv2D(filters=48, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        model = layers.Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        # model = layers.MaxPooling2D(pool_size=(2, 2))(model)

        model = layers.Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        model = layers.Conv2D(filters=32, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        # model = layers.MaxPooling2D(pool_size=(2, 2))(model)

        model = layers.Conv2D(filters=16, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        # ------------------------------------------------------------------------------------------------------
        model = layers.Conv2D(filters=128, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2DTranspose(filters=8, kernel_size=self.kernel_size, strides=2, padding='same',
                                       activation='relu')(model)

        model = layers.Conv2D(filters=128, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2DTranspose(filters=16, kernel_size=self.kernel_size, strides=2, padding='same',
                                       activation='relu')(model)

        model = layers.Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2DTranspose(filters=32, kernel_size=self.kernel_size, strides=2, padding='same',
                                       activation='relu')(model)

        model = layers.Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=32, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=16, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=8, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=3, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        model = layers.Lambda(lambda x: x * 255)(model)

        model = Model(inputs=[input_gray_pic], outputs=[model])

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
        model.add(layers.Conv2D(64, self.kernel_size, padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(64))
        model.add(layers.Dense(16))
        model.add(layers.Dense(1))

        model.summary()

        plot_model(model, to_file='networkStructure/discriminator.png', show_shapes=True, show_layer_names=True)

        return model
