from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


class Autoencoder:

    def __init__(self):
        self.kernel_size = (3, 3)
        self.encoder = self.createEncoder()
        self.decoder = self.createDecoder()

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def createEncoder(self):
        input_gray_pic = layers.Input(shape=(512, 512, 1))
        input_color_hue = layers.Input(shape=(10,))

        Hinput = layers.Lambda(lambda x: x/180)(input_color_hue)
        model = layers.Lambda(lambda x: x/255)(input_gray_pic)

        # CNN LAYERS
        model = layers.Conv2D(filters=8, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.MaxPooling2D(pool_size=(2, 2))(model)

        model = layers.Conv2D(filters=16, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.MaxPooling2D(pool_size=(2, 2))(model)

        model = layers.Conv2D(filters=24, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        model = layers.Conv2D(filters=32, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.MaxPooling2D(pool_size=(2, 2))(model)

        model = layers.Conv2D(filters=48, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        model = layers.Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.MaxPooling2D(pool_size=(2, 2))(model)

        model = layers.Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        model = layers.Conv2D(filters=32, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.MaxPooling2D(pool_size=(2, 2))(model)

        model = layers.Conv2D(filters=16, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=16, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        model = layers.Conv2D(filters=8, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=8, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        # FLATTEN
        model = layers.Flatten()(model)

        # DENSE
        model = layers.Dense(256, activation='relu')(model)
        model = layers.Dense(128, activation='relu')(model)
        model = layers.Dense(64, activation='relu')(model)
        model = layers.Dense(64, activation='relu')(model)

        combined = layers.Concatenate()([model, Hinput])
        model = layers.Dense(64, activation='relu')(combined)

        combined = layers.Concatenate()([model, Hinput])
        model = layers.Dense(64, activation='relu')(combined)

        combined = layers.Concatenate()([model, Hinput])
        model = layers.Dense(64, activation='relu')(combined)

        combined = layers.Concatenate()([model, Hinput])
        model = layers.Dense(64, activation='relu')(combined)

        combined = layers.Concatenate()([model, Hinput])
        model = layers.Dense(64, activation='relu')(combined)

        combined = layers.Concatenate()([model, Hinput])
        model = layers.Dense(64, activation='relu')(combined)

        combined = layers.Concatenate()([model, Hinput])
        model = layers.Dense(64, activation='relu')(combined)

        combined = layers.Concatenate()([model, Hinput])
        model = layers.Dense(64, activation='relu')(combined)

        model = layers.Dense(64, activation='relu')(model)
        model = layers.Dense(64, activation='relu')(model)

        model = layers.Dense(64)(model)  # NO ACTIVATION

        model = Model(inputs=[input_gray_pic, input_color_hue], outputs=[model])

        model.summary()

        plot_model(model, to_file='networkStructure/encoder.png', show_shapes=True, show_layer_names=True)
        return model

    def createDecoder(self):
        input_layer = layers.Input(shape=(64,))

        model = layers.Dense(64 * 64, activation='relu')(input_layer)
        model = layers.Reshape((64, 64, 1))(model)
        model = layers.Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2DTranspose(filters=8, kernel_size=self.kernel_size, strides=2, padding='same',
                                       activation='relu')(model)
        model = layers.Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2DTranspose(filters=16, kernel_size=self.kernel_size, strides=2, padding='same',
                                       activation='relu')(model)
        model = layers.Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2DTranspose(filters=32, kernel_size=self.kernel_size, strides=2, padding='same',
                                       activation='relu')(model)
        model = layers.Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=32, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=16, kernel_size=self.kernel_size, padding='same', activation='relu')(model)
        model = layers.Conv2D(filters=3, kernel_size=self.kernel_size, padding='same', activation='relu')(model)

        model = Model(inputs=[input_layer], outputs=[model])

        model.summary()

        plot_model(model, to_file='networkStructure/decoder.png', show_shapes=True, show_layer_names=True)

        return model
