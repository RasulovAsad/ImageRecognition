import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from utils.image_utils import load_images_from_folder, preprocess_images, split_data
import matplotlib.pyplot as plt

num_classes = 5

# Image preprocessing
def load_images_from_folder(folder):
    print('loading images')
    images = []
    labels = []
    for subfolder in os.listdir(folder):
        label = subfolder
        subfolder_path = os.path.join(folder, subfolder)
        for filename in os.listdir(subfolder_path):
            img = cv2.imread(os.path.join(subfolder_path, filename))
            if img is not None:
                images.append(img)
                labels.append(label)

    return images, labels

def preprocess_images(images):
    resized_images = [cv2.resize(img, (224, 224)) for img in images]
    preprocessed_images = np.array(resized_images, dtype=np.float32) / 255.0
    return preprocessed_images

data_folder = 'src\\resources\\flowers'
images, labels = load_images_from_folder(data_folder)
preprocessed_images = preprocess_images(images)

# Split data into training and validation sets
print('Splitting data into training and validation sets')
X_train, X_val, y_train, y_val = split_data(preprocessed_images, labels, test_size=0.2, random_state=42)

# Convert string labels to integer values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

# Convert integers to one-hot encoding
y_train_onehot = to_categorical(y_train_encoded, num_classes)
y_val_onehot = to_categorical(y_val_encoded, num_classes)

# Create the model
print('Creating the model')
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Create data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Scale pixel values
    rotation_range=20,  # Image rotation for data augmentation (optional)
    width_shift_range=0.2,  # Width shift (optional)
    height_shift_range=0.2,  # Height shift (optional)
    horizontal_flip=True,  # Image reflection (optional)
    fill_mode='nearest'  # Fill method for data augmentation (optional)
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Only scale pixel values for validation data

# Create generators for training and validation data
batch_size = 32

train_generator = train_datagen.flow(
    X_train, y_train_onehot,  # Use y_train_onehot
    batch_size=batch_size
)

val_generator = val_datagen.flow(
    X_val, y_val_onehot,  # Use y_val_onehot
    batch_size=batch_size
)

# Train the model using data generators
num_epochs = 10

history = model.fit(
    train_generator,  # Training data generator
    epochs=num_epochs,
    validation_data=val_generator  # Validation data generator
)

# Accuracy plot
plt.plot(history.history['accuracy'], label='Accuracy on training set')
plt.plot(history.history['val_accuracy'], label='Accuracy on test set')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('plots\\vgg\\accuracy_plot.png')  # Save accuracy plot

# Loss plot
plt.figure()  # Create a new figure for the next plot
plt.plot(history.history['loss'], label='Loss on the training set')
plt.plot(history.history['val_loss'], label='Test sample loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('vgg_loss_plot.png')  # Save loss plot

print('Saving the model')
model.save('vgg_model.keras')
