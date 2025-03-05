import tensorflow as tf
import pandas as pd
import numpy as np

def normalize_img(image, label):
    image = tf.cast(image, tf.float32)  
    image = image / 255.0  
    return image, label

def preprocess_data(data):
    images, labels = data
    dataset = tf.data.Dataset.from_tensor_slices((images, labels)) 
    dataset = dataset.map(normalize_img)
    dataset = dataset.batch(32) 
    return dataset

def load_dataset(name):
    if name == "mnist":
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    elif name == "fashion_mnist":
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    elif name == "cifar10":
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    return (train_images, train_labels), (test_images, test_labels)

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.GlobalAveragePooling2D(),
        
        tf.keras.layers.Dense(64, activation='relu'),
        
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_predict(dataset_name, train_data, test_data, epochs, save_model=False):
    train_dataset = preprocess_data(train_data)
    test_dataset = preprocess_data(test_data)

    model = build_model((train_data[0].shape[1], train_data[0].shape[2], 1))

    history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset) 

    accuracy = history.history['accuracy'][-1] 
    loss = history.history['loss'][-1] 

    predictions = model.predict(test_data[0])
    predicted_labels = np.argmax(predictions, axis=1)

    predicted_df = pd.DataFrame({"Real Label": test_data[1].flatten(),"Predicted Label": predicted_labels})

    if save_model:
        model.save(f"model_{dataset_name}.keras") 

    return accuracy, loss, predicted_df, model  

def load_model(model_path):
    return tf.keras.models.load_model(model_path)
