import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ================= CONFIGURATION =================
DATA_DIR = "data"                 # Created by prepare_dataset.py
IMG_SIZE = (224, 224)
BATCH_SIZE = 32                   # Safe for laptop RAM
EPOCHS = 10
LEARNING_RATE = 1e-4
# =================================================


def main():
    print("🚀 Setting up data generators...")

    # ================= DATA GENERATORS =================
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=["real", "fake"],   # Explicit mapping: Real=0, Fake=1
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=["real", "fake"],
        shuffle=False
    )

    print("Class indices:", train_generator.class_indices)
    # Expected: {'real': 0, 'fake': 1}

    # ================= MODEL =================
    print("🏗️ Building MobileNetV2 model...")

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SIZE + (3,)
    )

    # Freeze base model
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    # ================= COMPILE =================
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    model.summary()

    # ================= CALLBACKS =================
    os.makedirs("models", exist_ok=True)

    callbacks = [
        # FIXED: Changed from .keras to .h5 to prevent the "options" error
        ModelCheckpoint(
            "models/faceguard_phase2_best.h5", 
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # ================= TRAIN =================
    print("🔥 Starting training...")
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("✅ Training complete!")
    print("💾 Best model saved to models/faceguard_phase2_best.h5")


if __name__ == "__main__":
    main()