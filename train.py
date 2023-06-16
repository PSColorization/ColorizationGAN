from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
import os
import time
import numpy as np

from matplotlib import pyplot as plt
from keras.backend import mean as kerasmean
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

BUFFER_SIZE = 60000
BATCH_SIZE = 16
EPOCHS = 5000
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
mae = tf.keras.losses.MeanSquaredError()

gen_losses = [[] for i in range(EPOCHS)]
disc_losses = [[] for i in range(EPOCHS)]


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) - tf.random.uniform(shape=real_output.shape, maxval=0.1),
                              real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output) + tf.random.uniform(shape=fake_output.shape, maxval=0.1),
                              fake_output)
    total_loss = (real_loss + fake_loss) / 2
    return total_loss


def wasserstein_loss(y_true, y_pred):
    return kerasmean(y_true * y_pred)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
# @tf.function
def train_step(input_batch_x, output_batch_y, epoch_number):
    global gen_losses
    global disc_losses
    (gray_pic, hue), rgb_pic = input_batch_x, output_batch_y
    for _ in range(5):
        with tf.GradientTape() as tape:
            generated_images = generator([gray_pic, hue], training=True)

            # Calculate discriminator's predictions
            real_labels = tf.ones((BATCH_SIZE, 1))
            fake_labels = -tf.ones((BATCH_SIZE, 1))
            real_output = discriminator(rgb_pic, training=True)
            fake_output = discriminator(generated_images, training=True)

            # Calculate discriminator loss
            disc_loss = wasserstein_loss(real_labels, real_output) + wasserstein_loss(fake_labels, fake_output)
            disc_losses[epoch_number].append(disc_loss)
    # Update discriminator weights
    discriminator_gradients = tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    clip_value_min = -0.01
    clip_value_max = 0.01
    for layer in discriminator.layers:
        weights = layer.get_weights()
        weights = [tf.clip_by_value(w, clip_value_min, clip_value_max) for w in weights]
        layer.set_weights(weights)

    with tf.GradientTape() as gen_tape:
        generated_images = generator([gray_pic, hue], training=True)
        fake_output = discriminator((generated_images), training=True)
        gen_loss = wasserstein_loss(fake_labels, fake_output)
        gen_losses[epoch_number].append(gen_loss)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))


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

        print('Time for epoch {} is {} sec. Gen_loss: {} , Disc_loss: {}'.format(epoch + 1,
                                                                                 round(time.time() - start, 3),
                                                                                 round(gen_losses[epoch][-1].numpy(),
                                                                                       3),
                                                                                 round(disc_losses[epoch][-1].numpy()),
                                                                                 3))


def run(GANmodel, trainData):
    global generator, discriminator
    global generator_optimizer, discriminator_optimizer
    global checkpoint, checkpoint_prefix

    generator_optimizer = tf.keras.optimizers.Adam(0.001)
    discriminator_optimizer = tf.keras.optimizers.Adam(0.000001)
    generator = tf.keras.models.load_model(
        './nonFreezedTrainOutputs/b_kesildi2_generatorTrained_499e.h5')  # GANmodel.generator #
    discriminator = tf.keras.models.load_model(
        './nonFreezedTrainOutputs/b_kesildi2_discTrained_499e.h5')  # GANmodel.discriminator #

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
    plt.savefig("./losses/losses.png")