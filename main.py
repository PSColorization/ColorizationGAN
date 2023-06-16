from GAN import GAN
import dataProcessing_old
import train
from encoder import autoencoder_model
import tensorflow as tf

if __name__ == '__main__':
    mission = 'train'
    # Preparing model network structure
    # GANmodel = GAN()

    if mission == 'train':
        # Preparing train, validation np.array's
        train_path = "./TEST/only_npz/wave"  # "./train/"
        val_path = "./val/"

        # (X_train_gray, X_train_hue), y_train = dataProcessing.get_X_y(train_path)
        # (X_val_gray, X_val_hue), y_val = dataProcessing.get_X_y(val_path)

        # print(f"X_train_gray shape: {X_train_gray.shape}")
        # print(f"X_train_hue shape: {X_train_hue.shape}")
        # print(f"y_train shape: {y_train.shape}")

        # print(f"X_val_gray shape: {X_val_gray.shape}")
        # print(f"X_val_hue shape: {X_val_hue.shape}")
        # print(f"y_val shape: {y_val.shape}")

        # train.run(GANmodel=GANmodel, trainData=dataProcessing_old.get_X_y(train_path))
        # train.run(GANmodel=GANmodel, trainData=train_path)
        autoencoder_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_absolute_error')

        # Print the summary of the autoencoder
        autoencoder_model.summary()
        train_data = dataProcessing_old.get_X_y(train_path)
        print(train_data[0][0].shape)

        input_images = [train_data[0][0]]

        # Train the autoencoder model
        autoencoder_model.fit(input_images, train_data[1], batch_size=16, epochs=3000)
        autoencoder_model.save(f"aeModel/model5unet_wave_500_light" + str(300) + "e.h5")



