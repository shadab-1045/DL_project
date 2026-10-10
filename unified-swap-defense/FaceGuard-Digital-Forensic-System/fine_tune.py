import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ================= CONFIGURATION =================
DATA_DIR = "data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10                  
LEARNING_RATE = 1e-5         
# =================================================


def main():
    print("🚀 Starting Phase 2B: Fine-Tuning (Direct Mode)...")

    # ================= DATA GENERATORS =================
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=["real", "fake"],
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

    # ================= LOAD MODEL =================
    print("📂 Loading best Phase-2 model...")
    model = load_model("models/faceguard_phase2_best.h5")

    # ================= DIRECT UNFREEZE =================
    print("🔓 Unfreezing top layers directly...")

    # 1. Reset: Start by unfreezing EVERYTHING
    for layer in model.layers:
        layer.trainable = True

    # 2. Freeze the bottom layers
    # We keep the last 50 layers active (Head + Top MobileNet blocks)
    # Everything BEFORE the last 50 gets frozen
    cutoff = -50
    print(f"   - Total layers: {len(model.layers)}")
    print(f"   - Freezing bottom {len(model.layers) + cutoff} layers...")
    
    for layer in model.layers[:cutoff]:
        layer.trainable = False

    # 3. FORCE FREEZE BatchNorm Layers (Advanced Polish)
    print("   - 🛡️ Forcing BatchNormalization layers to stay frozen...")
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # ================= RE-COMPILE =================
    print("🔧 Re-compiling with very low learning rate...")
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
    callbacks = [
        ModelCheckpoint(
            "models/faceguard_phase2_finetuned.h5",
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
    print("🔁 Phase 2B: Fine-tuning upper CNN layers...")
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("✅ Fine-tuning complete!")
    print("💾 Final model saved as: models/faceguard_phase2_finetuned.h5")


if __name__ == "__main__":
    main()