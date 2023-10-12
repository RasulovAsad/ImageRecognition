import cv2
import numpy as np
from keras.models import load_model

# Загрузка обученной модели
model = load_model('model\\my_model.keras')

# Функция для загрузки и предобработки изображения
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, (224, 224))
        img = np.array(img, dtype=np.float32) / 255.0
        return img
    else:
        return None

# Функция для классификации изображения с помощью обученной модели
def classify_image(image_path):
    # Загрузка и предобработка изображения
    img = load_and_preprocess_image(image_path)
    if img is None:
        return "Image not found or cannot be loaded."

    # Выполнение инференса с помощью модели
    result = model.predict(np.expand_dims(img, axis=0))
    class_index = np.argmax(result)
    
    # Здесь предполагается, что у вас есть список классов, например:
    class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    
    # Получение метки класса
    predicted_class = class_labels[class_index]
    
    return predicted_class

# Пример использования функции для классификации изображения
image_path = 'resources\\test\\dsa.jpg'
predicted_class = classify_image(image_path)
print(f'The image is classified as: {predicted_class}')
