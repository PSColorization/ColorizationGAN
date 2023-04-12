from GAN import GAN
import dataProcessing
import train

if __name__ == '__main__':
    mission = 'train'
    # Preparing model network structure
    GANmodel = GAN()

    if mission == 'train':
        # Preparing train, validation np.array's
        train_path = "./TEST/sky_onlynpz"
        val_path = "./val/"

        # (X_train_gray, X_train_hue), y_train = dataProcessing.get_X_y(train_path)
        # (X_val_gray, X_val_hue), y_val = dataProcessing.get_X_y(val_path)

        # print(f"X_train_gray shape: {X_train_gray.shape}")
        # print(f"X_train_hue shape: {X_train_hue.shape}")
        # print(f"y_train shape: {y_train.shape}")

        # print(f"X_val_gray shape: {X_val_gray.shape}")
        # print(f"X_val_hue shape: {X_val_hue.shape}")
        # print(f"y_val shape: {y_val.shape}")

        train.run(GANmodel=GANmodel, trainData=dataProcessing.get_X_y(train_path))


