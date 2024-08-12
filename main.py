import tensorflow as tf
import tensorflow_datasets as tfds

data, metadata = tfds.load("fashion_mnist", as_supervised=True, with_info=True)

training_data, testing_data = data["train"], data["test"]

def normalize(image, label):                #function to change RGB values(0-255) to (0-1)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label

training_data = training_data.map(normalize)
testing_data = testing_data.map(normalize)

training_data = training_data.cache()
testing_data = testing_data.cache()