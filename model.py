import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16
from tensorflow.keras.metrics import Precision, Recall

# === Configuration ===
base_dir = "data"
train_dir = os.path.join(base_dir, "national_id")

img_height, img_width = 224, 224
batch_size = 32
num_epochs = 50
fine_tune_epochs = 20
val_split = 0.2
class_names = ['forged', 'originals']

# === Data Augmentation and Preparation ===
def prepare_data(train_dir, img_height, img_width, batch_size, val_split):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=val_split
    )

    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        classes=class_names,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_data = datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        classes=class_names,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    return train_data, val_data

# === Compute Class Weights ===
def compute_class_weights(train_data):
    labels = train_data.classes
    return dict(enumerate(class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )))

# === Model Definition ===
def create_model(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the convolutional base

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),  # Increased dropout rate
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=['accuracy', Precision(), Recall()])
    return model

# === Callbacks ===
def get_callbacks():
    checkpoint_cb = ModelCheckpoint(
        "document_forgery_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    earlystop_cb = EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6
    )
    
    return [checkpoint_cb, earlystop_cb, lr_scheduler]

# === Visualization Function ===
def plot_history(history, fine_tune=False):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy over Epochs' + (' (Fine-Tuning)' if fine_tune else ''))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Main Execution ===
train_data, val_data = prepare_data(train_dir, img_height, img_width, batch_size, val_split)
class_weights = compute_class_weights(train_data)

# Create and train the model
model = create_model((img_height, img_width, 3))
callbacks = get_callbacks()

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=num_epochs,
    class_weight=class_weights,
    callbacks=callbacks
)

# === Fine-tuning the Model ===
# Unfreeze some layers for fine-tuning
for layer in model.layers[-4:]:
    layer.trainable = True

# Recompile the model with a lower learning rate
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

# Continue training with fine-tuning
history_fine_tune = model.fit(
    train_data,
    validation_data=val_data,
    epochs=fine_tune_epochs,
    class_weight=class_weights,
    callbacks=callbacks
)

# Plot training history
plot_history(history)
plot_history(history_fine_tune, fine_tune=True)