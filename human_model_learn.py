import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop

path = 'human_dataset'


def create_data_generators():
    data_gen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2)

    train_dataset = data_gen.flow_from_directory(path,
                                                 target_size=(200, 200),
                                                 batch_size=3,
                                                 class_mode='binary',
                                                 subset='training')

    validation_dataset = data_gen.flow_from_directory(path,
                                                      target_size=(200, 200),
                                                      batch_size=3,
                                                      class_mode='binary',
                                                      subset='validation')
    return train_dataset, validation_dataset


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(learning_rate=0.001),
                  metrics=['accuracy'])
    return model


def train_model(model, train_dataset, validation_dataset):
    model.fit(train_dataset,
              steps_per_epoch=3,
              epochs=1,
              validation_data=validation_dataset)

    test_loss, test_acc = model.evaluate(validation_dataset, verbose=2)
    print('\nTest accuracy:', test_acc)


def main():
    train_dataset, validation_dataset = create_data_generators()
    print(train_dataset.class_indices)

    model = create_model()
    train_model(model, train_dataset, validation_dataset)


if __name__ == '__main__':
    main()