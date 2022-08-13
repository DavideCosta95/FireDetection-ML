import tensorflow as tf

# CUSTOMIZE PATH
root_dir_path = "C:/Users/Gren/Desktop/fire_detector/train"


image_size = (180, 180)
batch_size = 32
epochs = 30
model_name = "fire_detection_from_scratch"

def make_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Image augmentation block
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomZoom(0.2, width_factor=None, fill_mode='reflect', interpolation='bilinear', seed=None, fill_value=0.0)
        ]
    )
    x = data_augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    previous_block_activation = x  # Set aside residual
    for size in [128, 256, 512, 728]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
        # Project residual
        residual = tf.keras.layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
    x = tf.keras.layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)


def train_data():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        root_dir_path,
        validation_split=0.2,
        subset="training",
        seed=621,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        root_dir_path,
        validation_split=0.2,
        subset="validation",
        seed=621,
        image_size=image_size,
        batch_size=batch_size,
    )
    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)
    model = make_model(input_shape=image_size + (3,))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(f"{model_name}_{{epoch}}.h5"),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds, shuffle=True,
    )

    return model


model = train_data()
