import os
import numpy as np
import tensorflow as tf
from PIL import Image
from style_transfer_model import build_style_transfer_model

# Parameters
IMAGE_SIZE = (1000, 1000)  # height, width Consistent with data generation
BATCH_SIZE = 1
EPOCHS = 30
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "model/handwriting_model.h5"
TRAIN_IMAGES_PATH = "output/train_images"

def load_images(image_folder):
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png'))]
    images = []
    for file in image_files:
        img = Image.open(file).convert('RGB')
        img = img.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))  # Consistent resizing: width, height
        img = np.array(img) / 255.0
        images.append(img)

    return np.array(images)

def main():
    train_images = load_images(TRAIN_IMAGES_PATH)
    print(f"Shape of train_images before model.fit: {train_images.shape}")
    model = build_style_transfer_model(img_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)) # height width
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.MeanSquaredError()

    model.compile(optimizer=optimizer, loss=loss_fn)
    model.fit(train_images, train_images, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

    model.save(MODEL_SAVE_PATH)
    print("Model saved!")

if __name__ == '__main__':
    main()