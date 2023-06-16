
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
import os
import time
import numpy as np
from dataProcessing import get_X_y

from matplotlib import pyplot as plt

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

BUFFER_SIZE = 60000
BATCH_SIZE = 16
EPOCHS = 10
input_dim = (256, 256, 1)
num_examples_to_generate = 16

num_samples = 10000

generator = None
discriminator = None
generator_optimizer = None
discriminator_optimizer = None
checkpoint = None
checkpoint_prefix = None
cross_entropy = BinaryCrossentropy()
mae = tf.keras.losses.MeanAbsoluteError()

gen_losses = [[] for i in range(EPOCHS)]
disc_losses = [[] for i in range(EPOCHS)]

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) - tf.random.uniform(shape=real_output.shape, maxval=0.1), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output) + tf.random.uniform(shape=fake_output.shape, maxval=0.1), fake_output)
    total_loss = (real_loss + fake_loss) / 2
    return total_loss


def generator_loss(fake_output, real_y):
    real_y = tf.cast( real_y , 'float32' )
    return mae(fake_output, real_y)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
#@tf.function
def train_step(input_batch_x, output_batch_y, epoch_number):
    (gray_pic, hue), rgb_pic = input_batch_x, output_batch_y
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([gray_pic, hue], training=True)

        real_output = discriminator((rgb_pic), training=True)
        fake_output = discriminator((generated_images), training=True)

        gen_loss = generator_loss(generated_images, rgb_pic)

        disc_loss = discriminator_loss(real_output, fake_output)

        gen_losses[epoch_number].append(gen_loss)
        disc_losses[epoch_number].append(disc_loss)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset):
    for epoch in range(EPOCHS):
        start = time.time()

        for (gray_batch_x, hue_batch_x), output_batch_y in dataset:
            train_step((gray_batch_x, hue_batch_x), output_batch_y, epoch)

        # (gray_batch_x, hue_batch_x), output_batch_y = dataset()
        # print(type(gray_batch_x))
        # print(gray_batch_x.shape)


        # Save the model every 20 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            generator.save(f"nonFreezedTrainOutputs/generatorTrained_" + str(epoch) + "e.h5")
            discriminator.save(f"nonFreezedTrainOutputs/discTrained_" + str(epoch) + "e.h5")
        try:
            print('Time for epoch {} is {} sec. Gen_loss: {} , Disc_loss: {}'.format(epoch + 1,
                                                                                 time.time() - start,
                                                                                 gen_losses[epoch][-1].numpy(),
                                                                                 disc_losses[epoch][-1].numpy()))
        except:
            pass
def run(GANmodel, trainData):
    global generator, discriminator
    global generator_optimizer, discriminator_optimizer
    global checkpoint, checkpoint_prefix

    generator_optimizer = tf.keras.optimizers.Adam(0.001)
    discriminator_optimizer = tf.keras.optimizers.Adam(0.000001)
    generator = GANmodel.generator #tf.keras.models.load_model('./nonFreezedTrainOutputs/a_forest_road_generatorTrained_999e.h5')
    discriminator = GANmodel.discriminator #tf.keras.models.load_model('./nonFreezedTrainOutputs/a_forest_road_discTrained_999e.h5') #

    # freeze_index = len(generator.layers) - 4
    # for layer in generator.layers[:freeze_index]:
    #    layer.trainable = False

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)


    # gray_pic = tf.data.Dataset.from_tensor_slices(trainData[0][0]).batch(BATCH_SIZE)
    # hue = tf.data.Dataset.from_tensor_slices(trainData[0][1]).batch(BATCH_SIZE)
    # rgb_pic = tf.data.Dataset.from_tensor_slices(trainData[1]).batch(BATCH_SIZE)
    trainData = tf.data.Dataset.from_tensor_slices(trainData).batch(BATCH_SIZE)
    # get_next_batch = get_X_y(trainData, batch_size=BATCH_SIZE, num_samples=10000)

    # trainData = (gray_pic, hue), rgb_pic
    # train(get_next_batch)
    train(trainData)

    # Saving losses graphs
    plt.plot(list(range(EPOCHS)), [i[-1].numpy() for i in gen_losses], color="red")
    plt.plot(list(range(EPOCHS)), [i[-1].numpy() for i in disc_losses], color="blue")
    plt.savefig("./losses.png")
