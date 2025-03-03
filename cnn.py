# Importing necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,accuracy_score
import numpy as np

# Part 1 - Data Preprocessing

# Data Augmentation for Training
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.1)  # 10% for validation

# Training set (90% of total training data)
training_set = train_datagen.flow_from_directory('input/train',
                                                 target_size=(64, 64),
                                                 batch_size=16,  # Reduced batch size
                                                 class_mode='binary',
                                                 subset='training')

# Validation set (10% of total training data)
validation_set = train_datagen.flow_from_directory('input/train',
                                                   target_size=(64, 64),
                                                   batch_size=16,
                                                   class_mode='binary',
                                                   subset='validation')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory('input/test',
                                            target_size=(64, 64),
                                            batch_size=16,
                                            class_mode='binary',
                                            shuffle=False)  # Keep order for confusion matrix

# Part 2 - Building the CNN

# Initializing the CNN
cnn = tf.keras.models.Sequential()

# First Convolutional Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Second Convolutional Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Third Convolutional Layer (Newly Added)
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Fully Connected Layer
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Dropout to Prevent Overfitting
cnn.add(tf.keras.layers.Dropout(0.5))

# Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN
cnn.fit(x=training_set, validation_data=validation_set, epochs=50)

# Part 4 - Evaluating on Test Set

# Get true labels
y_true = test_set.classes  # Corrected from test_set.labels

# Predict on test set
y_pred = cnn.predict(test_set, verbose=0)
y_pred = (y_pred > 0.5).astype("int32")  # Convert probabilities to binary labels

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Visualize the confusion matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_set.class_indices.keys())
cm_display.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Test Set")
plt.show()