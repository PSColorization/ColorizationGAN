import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, RepeatVector
from tensorflow.keras.layers import Input, Conv2DTranspose, BatchNormalization, Activation, Reshape, LeakyReLU, \
    MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import UpSampling2D


# Create the encoder using ResNet-50
def create_encoder2():
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    print("RESNETLENGTH: ", len(resnet.layers))
    # Freeze the first layers
    for layer in resnet.layers[:50]:
        layer.trainable = False
    encoder_output = resnet.output
    return Model(resnet.input, encoder_output)


# Create the decoder
def create_decoder():
    input_shape = (128, 128, 3)
    # color_palette_input = Input(shape=(256,256,10))
    inputs = Input(shape=input_shape)

    e1 = Conv2D(16, (3, 3), padding='same')(inputs)  # 256x256x16
    e1 = LeakyReLU(alpha=0.1)(e1)

    # ez = Conv2D(16, (3, 3), padding='same')(e1) #256x256x16
    # ez = LeakyReLU(alpha=0.1)(ez)
    # ez = BatchNormalization()(ez)

    # ez = concatenate([ez,color_palette_input],axis = -1) #256x256x32

    e2 = Conv2D(32, (3, 3), padding='same')(e1)  # 256x256x32
    e2 = BatchNormalization()(e2)
    e2 = LeakyReLU(alpha=0.1)(e2)

    en = Conv2D(64, (3, 3), padding='same', strides=(2, 2))(e2)  # 256x256x54
    en = BatchNormalization()(en)
    en = LeakyReLU(alpha=0.1)(en)

    # en = concatenate([en,color_palette_input],axis = -1) #256x256x64

    e3 = Conv2D(128, (3, 3), padding='same', strides=(2, 2))(en)  # 128x128x64
    e3 = BatchNormalization()(e3)
    e3 = LeakyReLU(alpha=0.1)(e3)

    e3 = Dropout(0.25)(e3)

    # e4 = Conv2D(64, (3, 3), padding='same',strides = (2,2))(e3) #64x64x64

    e5 = Conv2D(256, (3, 3), padding='same')(e3)  # 64x64x128
    e5 = BatchNormalization()(e5)
    e5 = LeakyReLU(alpha=0.1)(e5)
    e5 = Dropout(0.25)(e5)

    # e9 = Conv2D(256, (3, 3), padding='same')(e5) #64x64x256
    # e9 = BatchNormalization()(e9)
    # e9 = LeakyReLU(alpha=0.1)(e9)
    # e9 = Dropout(0.25)(e9)

    e11 = Conv2D(512, (3, 3), padding='same')(e5)  # 64x64x512
    e11 = BatchNormalization()(e11)
    e11 = LeakyReLU(alpha=0.1)(e11)

    e12 = Conv2D(1024, (3, 3), padding='same')(e11)  # 64x64x1024
    e12 = LeakyReLU(alpha=0.1)(e12)

    # encoder_model = tf.keras.models.Model([inputs], e12, name='encoder')

    # input_shape = (64, 64, 1024)  # Shape of the encoder output
    # decoder_input = Input(shape=input_shape)

    x1 = Conv2DTranspose(1024, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(e12)  # 64x64x1024

    x1 = concatenate([x1, e12], axis=-1)

    x2 = Conv2DTranspose(512, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x1)  # 64x64x512

    x2 = concatenate([x2, e11], axis=-1)

    x3 = Conv2DTranspose(256, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x2)  # 64x64x256

    x3 = concatenate([x3, e5], axis=-1)

    xu1 = UpSampling2D(size=(2, 2))(x3)

    # x4 = Conv2DTranspose(256, kernel_size=(3, 3),activation = 'relu' ,strides=(1, 1), padding='same')(xu1) #128x128x256

    x5 = Conv2DTranspose(128, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(xu1)  # 128x128x128

    # x6 = Conv2DTranspose(64, kernel_size=(3, 3),activation = tf.keras.layers.LeakyReLU(alpha=0.1) ,strides=(1, 1), padding='same')(x5) #128x128x64

    x7 = Conv2DTranspose(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x5)  # 128x128x64

    # xu2 = UpSampling2D(size=(2, 2))(x7)

    x8 = Conv2DTranspose(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x7)  # 256x256x64

    x8 = concatenate([x8, en], axis=-1)  # 256x256x64

    xu2 = UpSampling2D(size=(2, 2))(x7)

    x9 = Conv2DTranspose(32, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(xu2)  # 256x256x32

    xt = Conv2DTranspose(32, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x9)  # 256x256x32

    xt = concatenate([xt, e2], axis=-1)

    # x10 = Conv2DTranspose(16, kernel_size=(3, 3),activation = 'relu' ,strides=(1, 1), padding='same')(xt) #256x256x16

    # x10 = concatenate([x10,e1], axis = -1)

    x11 = Conv2DTranspose(16, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(xt)  # 256x256x16

    x11 = concatenate([x11, e1], axis=-1)

    decoder_output = Conv2DTranspose(3, kernel_size=(3, 3), activation='relu', padding='same')(x11)  # 256x256x3

    conc = concatenate([decoder_output, inputs], axis=-1)

    decoder_ouput = Conv2D(3, (3, 3), padding='same')(conc)

    return tf.keras.models.Model([inputs], decoder_output, name='ae')


def create_encoder():
    input_shape = (256, 256, 3)
    color_palette_input = Input(shape=(64, 64, 10))
    inputs = Input(shape=input_shape)

    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), padding='same', strides=(2, 2))(x)
    x = concatenate([x, color_palette_input], axis=-1)

    z = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(z)
    x = LeakyReLU(alpha=0.1)(x)

    x = concatenate([x, z], axis=-1)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)

    y = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(y)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = concatenate([x, y], axis=-1)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(1024, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)

    encoder_model = tf.keras.models.Model([inputs, color_palette_input], x, name='encoder')
    return encoder_model


def create_autoencoder(decoder):
    decoder_input = decoder.input

    decoder_output = decoder(decoder_input)

    return Model([decoder_input], decoder_output)


# encoder_model = create_encoder()
decoder_model = create_decoder()
autoencoder_model = create_autoencoder(decoder_model)


