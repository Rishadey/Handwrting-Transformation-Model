import tensorflow as tf

def residual_block(x, filters):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut])
    return x


def build_style_transfer_model(img_shape=(256, 128, 3)):  # Consistent with data generation (height, width, channels)
    inputs = tf.keras.layers.Input(shape=img_shape)

    # Encoding
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Residual blocks
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)


    # Decoding
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    outputs = tf.keras.layers.Conv2D(3, kernel_size=3, padding='same', activation='linear')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == '__main__':
    model = build_style_transfer_model()
    model.summary()