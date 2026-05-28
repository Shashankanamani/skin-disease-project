import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow

# =========================
# FIX RANDOMNESS
# =========================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("Setup Done ")

# =========================
# MLFLOW SETUP
# =========================

mlflow.set_experiment("Skin Disease Detection")

mlflow.start_run()

# =========================
# DATASET SETTINGS
# =========================

DATASET_PATH = "dataset"

IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 8

# =========================
# LOAD DATASET
# =========================

data = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

class_names = data.class_names

print("Classes:", class_names)

# =========================
# NORMALIZE
# =========================

data = data.map(lambda x, y: (x / 255.0, y))

# =========================
# SPLIT DATA
# =========================

train_size = int(0.8 * len(data))

train_data = data.take(train_size)

test_data = data.skip(train_size)

print("Data Ready")

# =========================
# BUILD MODEL
# =========================

model = tf.keras.Sequential([

    tf.keras.Input(shape=(128,128,3)),

    tf.keras.layers.Conv2D(
        32,
        (3,3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        len(class_names),
        activation='softmax'
    )
])

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# MODEL SUMMARY
# =========================

model.summary()

# =========================
# LOG PARAMETERS
# =========================

mlflow.log_param("Image Size", IMG_SIZE)
mlflow.log_param("Batch Size", BATCH_SIZE)
mlflow.log_param("Epochs", EPOCHS)

# =========================
# TRAIN MODEL
# =========================

history = model.fit(
    train_data,
    epochs=EPOCHS
)

# =========================
# EVALUATE MODEL
# =========================

test_loss, test_acc = model.evaluate(test_data)

print("Base Accuracy:", test_acc)

# =========================
# LOG METRICS
# =========================

mlflow.log_metric("Accuracy", float(test_acc))
mlflow.log_metric("Loss", float(test_loss))

# =========================
# PLOT ACCURACY
# =========================

plt.plot(history.history['accuracy'])

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.savefig("accuracy_plot.png")

plt.show()

mlflow.log_artifact("accuracy_plot.png")

# =========================
# PRUNING
# =========================

def prune_weights(model, pruning_percent=0.7):

    new_weights = []

    for layer in model.layers:

        weights = layer.get_weights()

        if len(weights) > 0:

            w = weights[0]

            threshold = np.percentile(
                np.abs(w),
                pruning_percent * 100
            )

            w[np.abs(w) < threshold] = 0

            weights[0] = w

        new_weights.append(weights)

    for i, layer in enumerate(model.layers):

        if len(new_weights[i]) > 0:

            layer.set_weights(new_weights[i])

    return model

# =========================
# APPLY PRUNING
# =========================

pruned_model = prune_weights(model, 0.7)

loss, acc = pruned_model.evaluate(test_data)

print("After Pruning Accuracy:", acc)

mlflow.log_metric("Pruned Accuracy", float(acc))

# =========================
# PLOT PRUNING
# =========================

plt.plot(
    [test_acc, acc],
    marker='o'
)

plt.xticks(
    [0,1],
    ['Original','Pruned']
)

plt.title("Accuracy After Pruning")

plt.ylim(0,1)

plt.savefig("pruning_plot.png")

plt.show()

mlflow.log_artifact("pruning_plot.png")

# =========================
# QUANTIZATION
# =========================

converter = tf.lite.TFLiteConverter.from_keras_model(
    pruned_model
)

tflite_model = converter.convert()

with open("model_quant.tflite", "wb") as f:

    f.write(tflite_model)

print("Quantized Model Saved ")
mlflow.log_artifact("model_quant.tflite")

# =========================
# SAVE MODEL
# =========================

model.save("model.h5")

print("Model Saved ")

mlflow.tensorflow.log_model(
    model,
    "model"
)

mlflow.log_artifact("model.h5")

mlflow.end_run()

print("MLflow Tracking Completed ")