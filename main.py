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

model = tf.keras.Sequential({
    
  tf.keras.layers.Conv2D(32, (3,3), input_shape=(28, 28, 1), activation="relu"),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(units=100, activation="relu"),
  tf.keras.layers.Dense(units=10, activation="softmax")

})

model.compile(
    
  optimizer="adam",
  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
  metrics=["accuracy"]

)

BATCH_SIZE = 32

training_data = training_data.repeat().shuffle(metadata.splits["train"].num_examples).batch(BATCH_SIZE)
testing_data = testing_data.batch(BATCH_SIZE)

historical = model.fit(training_data, epochs=5)