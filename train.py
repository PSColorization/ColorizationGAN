from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
import os
import time
import numpy as np

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

BUFFER_SIZE = 60000
BATCH_SIZE = 16
EPOCHS = 100
input_dim = (256, 256, 1)
num_examples_to_generate = 16

generator = None
discriminator = None
generator_optimizer = None
discriminator_optimizer = None
checkpoint = None
checkpoint_prefix = None
cross_entropy = BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
# @tf.function
def train_step(input_batch_x, output_batch_y):
    (gray_pic, hue), rgb_pic = input_batch_x, output_batch_y
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([gray_pic, hue], training=True)
        real_output = discriminator(rgb_pic, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)

        disc_loss = discriminator_loss(real_output, fake_output)



    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return (gen_loss, disc_loss)

def train(dataset):
    for epoch in range(EPOCHS):
        start = time.time()

        for (gray_batch_x, hue_batch_x), output_batch_y in dataset:
            (gen_loss, disc_loss) = train_step((gray_batch_x, hue_batch_x), output_batch_y)
            print((gen_loss, disc_loss))

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            generator.save("generatorTrained.h5")

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


def run(GANmodel, trainData):
    global generator, discriminator
    global generator_optimizer, discriminator_optimizer
    global checkpoint, checkpoint_prefix

    generator_optimizer = tf.keras.optimizers.Adam(0.001)
    discriminator_optimizer = tf.keras.optimizers.Adam(0.000001)
    generator = GANmodel.generator
    discriminator = GANmodel.discriminator

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)


    trainData = tf.data.Dataset.from_tensor_slices(trainData).batch(BATCH_SIZE)

    train(trainData)
