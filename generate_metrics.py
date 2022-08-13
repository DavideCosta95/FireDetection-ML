import numpy
import tensorflow as tf
from sklearn.metrics import classification_report
import tensorflow_hub as hub

# CUSTOMIZE PATHS
test_dir_fire_path = "C:/Users/Gren/Desktop/fire_detector/test/fire"
test_dir_nofire_path = "C:/Users/Gren/Desktop/fire_detector/test/not_fire"
model_path = "C:/Users/Gren/Desktop/fire_detection/fire_detection.h5"


image_size = (180, 180)
batch_size = 32
epochs = 30
model_name = "fire_detection"


def perform_test(model):
    fire_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir_fire_path,
        seed=621,
        image_size=image_size,
        batch_size=batch_size,
    )
    predictions_fire = model.predict(fire_test_ds)

    for i in range(predictions_fire.size):
        if predictions_fire[i] >= 0.50:
            predictions_fire[i] = 0
        else:
            predictions_fire[i] = 1

    nofire_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir_nofire_path,
        seed=621,
        image_size=image_size,
        batch_size=batch_size,
    )
    predictions_nofire = model.predict(nofire_test_ds)

    for i in range(predictions_nofire.size):
        if predictions_nofire[i] >= 0.50:
            predictions_nofire[i] = 0
        else:
            predictions_nofire[i] = 1

    y_true = []
    for i in range(predictions_fire.size):
        y_true.append(1)

    for i in range(predictions_nofire.size):
        y_true.append(0)

    y_pred = numpy.concatenate((predictions_fire, predictions_nofire), axis=0)

    print(classification_report(y_true, y_pred))


def run_test():
    model = tf.keras.models.load_model(model_path,
                                       custom_objects={'KerasLayer': hub.KerasLayer})
    perform_test(model)


run_test()
