import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# Define image size and batch size
img_size = 48
batch_size = 64

# ImageDataGenerator for augmenting training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# No augmentation for test data, just rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data from the directory
train_data = train_datagen.flow_from_directory(
    'D:\\ML_Projects\\Dataset\\train',  # This is the path to your train folder
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

# Load test data from the directory
test_data = test_datagen.flow_from_directory(
    r'D:\ML_Projects\Dataset\test',  # This is the path to your test folder
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)


# Building the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(7, activation='softmax')  # 7 emotion categories
])

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Model summary to check architecture
model.summary()




# Train the model
history = model.fit(
    train_data,
    epochs=25,
    validation_data=test_data
)

# Save the trained model
model.save('emotion_detection_model.h5')
