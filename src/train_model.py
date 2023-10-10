import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

num_classes = 5

# Предобработка изображений
def load_images_from_folder(folder):
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

# Разделение данных на обучающую и проверочную выборки
print('Разделение данных на обучающую и проверочную выборки')
X_train, X_val, y_train, y_val = train_test_split(preprocessed_images, labels, test_size=0.2, random_state=42)

# Создание модели
print('Создание модели')
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Создайте генераторы данных для обучения и проверки
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Масштабирование значений пикселей
    rotation_range=20,  # Вращение изображений для аугментации данных (по желанию)
    width_shift_range=0.2,  # Сдвиг по ширине (по желанию)
    height_shift_range=0.2,  # Сдвиг по высоте (по желанию)
    horizontal_flip=True,  # Отражение изображений (по желанию)
    fill_mode='nearest'  # Метод заполнения при аугментации данных (по желанию)
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Только масштабирование для проверочных данных

# Создайте генераторы для обучающих и проверочных данных
batch_size = 32

train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=batch_size
)

val_generator = val_datagen.flow(
    X_val, y_val,
    batch_size=batch_size
)

# Обучение модели с использованием генераторов данных
num_epochs = 10

history = model.fit(
    train_generator,  # Генератор обучающих данных
    epochs=num_epochs,
    validation_data=val_generator  # Генератор проверочных данных
)

# # Обучение модели
# print('Обучение модели')
# num_epochs = 10

# history = model.fit(
#     X_train, y_train,
#     epochs=num_epochs,
#     validation_data=(X_val, y_val)
# )
print('Сохранение модели')
model.save('my_model.h5')
