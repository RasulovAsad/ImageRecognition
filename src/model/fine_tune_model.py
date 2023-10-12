from keras.models import load_model
from keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from utils.image_utils import load_images_from_folder, preprocess_images, split_data

# Загрузка обученной модели
model = load_model('my_model.h5')

# Заморозка верхних слоев
for layer in model.layers[:-5]:  # Заморозить все слои кроме последних пяти
    layer.trainable = False

# Добавление новых слоев для задачи классификации
x = model.layers[-2].output
num_classes = 5  # Количество классов в вашей задаче
predictions = Dense(num_classes, activation='softmax')(x)
new_model = Model(inputs=model.input, outputs=predictions)

# Настройка оптимизатора и параметров обучения
new_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Загрузка и масштабирование данных
data_folder = 'path_to_your_data_folder'
images, labels = load_images_from_folder(data_folder)
preprocessed_images = preprocess_images(images)

# Разделение данных на обучающую и проверочную выборки
X_train, X_val, y_train, y_val = split_data(preprocessed_images, labels, test_size=0.2, random_state=42)

# Преобразование строковых меток в целые числа
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

# Преобразование целых чисел в one-hot кодировку
y_train_onehot = to_categorical(y_train_encoded, num_classes)
y_val_onehot = to_categorical(y_val_encoded, num_classes)

# Создание генераторов данных для обучения и проверки
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Масштабирование значений пикселей
    rotation_range=20,  # Вращение изображений для аугментации данных (по желанию)
    width_shift_range=0.2,  # Сдвиг по ширине (по желанию)
    height_shift_range=0.2,  # Сдвиг по высоте (по желанию)
    horizontal_flip=True,  # Отражение изображений (по желанию)
    fill_mode='nearest'  # Метод заполнения при аугментации данных (по желанию)
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Только масштабирование для проверочных данных

# Создание генераторов для обучающих и проверочных данных
batch_size = 32

train_generator = train_datagen.flow(
    X_train, y_train_onehot,
    batch_size=batch_size
)

val_generator = val_datagen.flow(
    X_val, y_val_onehot,
    batch_size=batch_size
)

# Обучение модели с использованием генераторов данных
num_epochs = 10

history = new_model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator
)

# Сохранение доученной модели
new_model.save('fine_tuned_model.h5')
