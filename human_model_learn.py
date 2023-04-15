import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers.rmsprop import RMSprop
import matplotlib.pyplot as plt
import numpy as np


path = 'human_dataset'

train = ImageDataGenerator(rescale=1/255, validation_split=0.2)
train_dataset = train.flow_from_directory(path,
                                          target_size=(200,200),
                                          batch_size=3,
                                          class_mode='binary',
                                          subset='training')

validation_dataset = train.flow_from_directory(path,
                                          target_size=(200,200),
                                          batch_size=3,
                                          class_mode='binary',
                                          subset='validation')

print(train_dataset.class_indices)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation= 'relu', input_shape= (200,200,3)),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(32, (3,3), activation= 'relu', input_shape= (200,200,3)),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(64, (3,3), activation= 'relu', input_shape= (200,200,3)),
    tf.keras.layers.MaxPool2D(2,2),
    ##
    tf.keras.layers.Flatten(),
    ##
    tf.keras.layers.Dense(512, activation= 'relu'),
    ##
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer= RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

model.fit = model.fit(train_dataset,
                      steps_per_epoch=3,
                      epochs=1,
                      validation_data=validation_dataset
                      )


# img = tf.keras.preprocessing.image.load_img('human_dataset/nothing/1.png', target_size=(200,200))
# # plt.imshow(img)
# # plt.show()

# x = tf.keras.preprocessing.image.img_to_array(img) / 255.0
# x = np.expand_dims(x,axis=0)
# images = np.vstack([x])
# val = model.predict(images)
# print(val)
# # if val == 0:

# # else:
# #     print('nie dziala')

test_loss, test_acc = model.evaluate(validation_dataset, verbose=2)
print('\nTest accuracy:', test_acc)




# model.save('ripo_model')