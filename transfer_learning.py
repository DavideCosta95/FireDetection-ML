import tensorflow as tf
from tensorflow.keras import applications
import matplotlib.pyplot as plt


# CUSTOMIZE PATH
root_dir_path = "C:/Users/Gren/Desktop/fire_detector/train"


image_size = (180, 180)
batch_size = 32
epochs = 30
model_name = "fire_detection_TL"


def get_mixed_model(input_shape):
    model = applications.VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    for layer in model.layers:
        layer.trainable = False

    inputs = model.inputs

    x = model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)


def train_data():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        root_dir_path,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        root_dir_path,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)
    model = get_mixed_model(input_shape=image_size + (3,))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(f"{model_name}_{{epoch}}.h5"),
    ]
    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    network_history = model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    )

    plot_history(network_history)


def plot_history(network_history):
    x_plot = list(range(1, epochs + 1))
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(x_plot, network_history.history['loss'])
    plt.plot(x_plot, network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(x_plot, network_history.history['accuracy'])
    plt.plot(x_plot, network_history.history['val_accuracy'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()


train_data()
