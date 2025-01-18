pip install gradio
pip install tensorflow
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import gradio as gr
import zipfile

driver_path = '/content/Driver_Activity.zip'
driver_folder_path = '/content/driver_behavior_dataset'

with zipfile.ZipFile(driver_path, 'r') as zip_ref:
    zip_ref.extractall(driver_folder_path)

driver_files = os.listdir(driver_folder_path)
print("Extracted files:", driver_files)
Drivers_Fr_path= os.path.join(driver_folder_path, 'Driver_Activity')
Drivers_Fr_files = os.listdir(Drivers_Fr_path)

print("Contents of 'bone_fracture' folder:", Drivers_Fr_files)
Drivers_Fr_files_path= os.path.join(Drivers_Fr_path, 'Driving Dataset')
Drivers_Fr_files_files = os.listdir(Drivers_Fr_files_path)

print("Contents of 'bone_fracture' folder:", Drivers_Fr_files_files)
train_driver = os.path.join(Drivers_Fr_files_path, 'Training')
test_driver = os.path.join(Drivers_Fr_files_path, 'Testing')

train_files = os.listdir(train_driver)
test_files = os.listdir(test_driver)

print("Training files:", train_files)
print("Testing files:", test_files)
DriverBehavior = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

DriverBehavior.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1.0/255.0, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_driver,
    batch_size=32,
    class_mode='categorical',
    target_size=(150, 150)
)

test_generator = test_datagen.flow_from_directory(
    test_driver,
    batch_size=32,
    class_mode='categorical',
    target_size=(150, 150)
)
driver_behavior_model = DriverBehavior.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // 32,
    verbose=2
)
def classify_driver_behavior(img):
    img = img.resize((150, 150))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = img_arr / 255.0  # Normalize

    prediction = DriverBehavior.predict(img_arr)
    behavior_labels = ['Safe Driving', 'talking_phone']
    predicted_behavior = behavior_labels[np.argmax(prediction)]

    return f"Predicted Behavior: {predicted_behavior}"
driver_interface = gr.Interface(
    fn=classify_driver_behavior,
    inputs=gr.Image(type="pil"),
    outputs=gr.Text(),
    live=True,
    title="Driver Behavior Identifier",
    description="Upload a driver image, and the model will classify it into behaviors like 'Safe Driving', 'talking_phone'"
)

driver_interface.launch()
