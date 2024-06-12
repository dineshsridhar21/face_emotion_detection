from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt

train_data_emotion = ImageDataGenerator(rescale=1./255)

train_generator = train_data_emotion.flow_from_directory(
    r'C:\Users\Dinesh\Office_work\face_emotion_detection\emotion_data\train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical"
)

test_generator = train_data_emotion.flow_from_directory(
    r'C:\Users\Dinesh\Office_work\face_emotion_detection\emotion_data\test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical"
)

emotion_model = Sequential()
emotion_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))  # Update input_shape
emotion_model.add(MaxPooling2D((2, 2)))
emotion_model.add(Conv2D(64, (3, 3), activation='relu'))
emotion_model.add(MaxPooling2D((2, 2)))
emotion_model.add(Conv2D(128, (3, 3), activation='relu'))
emotion_model.add(MaxPooling2D((2, 2)))
emotion_model.add(Flatten())
emotion_model.add(Dense(128, activation='relu'))
emotion_model.add(Dense(7, activation='softmax'))

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001, decay=1e-6), metrics=['accuracy'])

history = emotion_model.fit(train_generator,
                    steps_per_epoch= 28709 // 64,
                    epochs = 40,
                    validation_data= test_generator,
                    validation_steps= 7178//64)

emotion_model.save('emotion_model_weights.h5')


history_detect = emotion_model.history.history
history_df = pd.DataFrame(history_detect)

plt.plot(history_df['loss'], label='Training Loss')
plt.title('training Loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.show

plt.plot(history_df['accuracy'], label='Training Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.show()
