import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    Add,
    Embedding
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

DATASET_DIR = "dataset"

X = []
y = []

labels = sorted(os.listdir(DATASET_DIR))
label_map = {label: idx for idx, label in enumerate(labels)}

for label in labels:
    folder = os.path.join(DATASET_DIR, label)

    for file in os.listdir(folder):
        if file.endswith(".npy"):
            X.append(np.load(os.path.join(folder, file)))
            y.append(label_map[label])

X = np.array(X, dtype=np.float32)
y = np.array(y)

print(X.shape)
print(y.shape)
print("Classes:", len(np.unique(y)))

num_classes = len(labels)

X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

y_train = to_categorical(y_train_raw, num_classes)
y_test = to_categorical(y_test_raw, num_classes)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_raw),
    y=y_train_raw
)

class_weights = dict(enumerate(class_weights))

sequence_length = X.shape[1]
feature_dim = X.shape[2]

inputs = Input(shape=(sequence_length, feature_dim))

x = Dense(128)(inputs)

positions = tf.range(
    start=0,
    limit=sequence_length,
    delta=1
)

position_embedding = Embedding(
    input_dim=sequence_length,
    output_dim=128
)(positions)

x = x + position_embedding

attn_output = MultiHeadAttention(
    num_heads=4,
    key_dim=32,
    dropout=0.2
)(x, x)

x = Add()([x, attn_output])
x = LayerNormalization()(x)

ffn = Dense(256, activation="relu")(x)
ffn = Dropout(0.2)(ffn)
ffn = Dense(128)(ffn)

x = Add()([x, ffn])
x = LayerNormalization()(x)

attn_output = MultiHeadAttention(
    num_heads=4,
    key_dim=32,
    dropout=0.2
)(x, x)

x = Add()([x, attn_output])
x = LayerNormalization()(x)

ffn = Dense(256, activation="relu")(x)
ffn = Dropout(0.2)(ffn)
ffn = Dense(128)(ffn)

x = Add()([x, ffn])
x = LayerNormalization()(x)

x = GlobalAveragePooling1D()(x)

x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)

outputs = Dense(
    num_classes,
    activation="softmax"
)(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    class_weight=class_weights,
    callbacks=[
        early_stop,
        reduce_lr
    ]
)

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"Test Accuracy: {accuracy:.4f}")

model.save("transformer_model_v3.keras")