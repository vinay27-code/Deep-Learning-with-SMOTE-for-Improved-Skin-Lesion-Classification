# === 📦 Imports ===
import cv2

# Install missing packages if needed
!pip install -q opencv-python-headless tensorflow

# Standard libraries
import os
import math
import gc
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from collections import Counter
from functools import partial

# OpenCV
import cv2

# TensorFlow / Keras (Colab-compatible)
from tensorflow.keras import layers, backend as K
from tensorflow.keras.applications import (
    ResNet50, MobileNet, DenseNet201, InceptionV3,
    NASNetLarge, InceptionResNetV2, NASNetMobile
)
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn import metrics

# Scipy
import scipy


from google.colab import drive
drive.mount('/content/drive')

from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

def build_model(base_model, lr=1e-4):
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)  # Change 5 to number of classes

    model = models.Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# Create validation generator
val_generator = ImageDataGenerator(rescale=1./255)
val_data = val_generator.flow(x_val, y_val, batch_size=BATCH_SIZE)

# Train the model
history = model.fit(
    train_data,  # from earlier: train_data = train_generator.flow(...)
    steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
    epochs=10,
    validation_data=val_data,
    callbacks=[learn_control, checkpoint],
    validation_steps=x_val.shape[0] // BATCH_SIZE
)

import matplotlib.pyplot as plt

history_df = pd.DataFrame(history.history)
history_df[['accuracy', 'val_accuracy']].plot(figsize=(8, 5))
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_val, Y_val_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Compute and plot confusion matrix (for binary labels: 0 = benign, 1 = malignant)
cm = confusion_matrix(Y_test, Y_pred)
cm_plot_labels = ['benign', 'malignant']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix for Skin Cancer')

from sklearn.metrics import confusion_matrix

# Convert TTA predictions to binary
Y_pred_tta_binary = (Y_pred_tta > 0.5).astype("int32")

# Compute confusion matrix directly using binary labels
cm = confusion_matrix(Y_test, Y_pred_tta_binary)

cm_plot_label = ['benign', 'malignant']
plot_confusion_matrix(cm, cm_plot_label, title='Confusion Matrix for Skin Cancer (TTA)')

from sklearn.metrics import classification_report

# Convert TTA predictions to binary
Y_pred_tta_binary = (Y_pred_tta > 0.5).astype("int32")

# Generate classification report
print(classification_report(Y_test, Y_pred_tta_binary, target_names=['benign', 'malignant']))

from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# Use raw probabilities for ROC (not thresholded)
roc_log = roc_auc_score(Y_test, Y_pred_tta)

false_positive_rate, true_positive_rate, threshold = roc_curve(Y_test, Y_pred_tta)
area_under_curve = auc(false_positive_rate, true_positive_rate)

# Plot ROC Curve
plt.plot([0, 1], [0, 1], 'r--')
plt.plot(false_positive_rate, true_positive_rate, label='AUC = {:.3f}'.format(area_under_curve))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# === 📁 Dataset Loading ===
def Dataset_loader(DIR, RESIZE, sigmaX=10):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR,IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".jpg":
            img = read(PATH)

            img = cv2.resize(img, (RESIZE,RESIZE))

            IMG.append(np.array(img))
    return IMG

benign_train = np.array(Dataset_loader('/content/drive/MyDrive/TESTING/benign/train',224))
malign_train = np.array(Dataset_loader('/content/drive/MyDrive/TESTING/malignant/train',224))
benign_test = np.array(Dataset_loader('/content/drive/MyDrive/TESTING/benign/test',224))
malign_test = np.array(Dataset_loader('/content/drive/MyDrive/TESTING/malignant/test',224))

# === ⚙️ Preprocessing ===
x_train, x_val, y_train, y_val = train_test_split(
    X_train, Y_train,
    test_size=0.2,
    random_state=11
)

BATCH_SIZE = 50

train_generator = ImageDataGenerator(
    rescale=1./255,
    zoom_range=2,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True
)

train_data = train_generator.flow(x_train, y_train, batch_size=BATCH_SIZE)

# === 🏋️ Training ===
learn_control = ReduceLROnPlateau(monitor='val_accuracy', patience=5,
                                  verbose=1,factor=0.2, min_lr=1e-7)

filepath = 'weights.best.weights.h5'

checkpoint = ModelCheckpoint(
    filepath=filepath,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
    mode='max'
)

# === 🖼️ Visualization ===
w = 60
h = 40
fig = plt.figure(figsize=(15, 15))
columns = 4
rows = 3

for i in range(1, columns * rows + 1):
    ax = fig.add_subplot(rows, columns, i)
    if np.argmax(y_train[i]) == 0:
        ax.title.set_text('Benign')
    else:
        ax.title.set_text('Malignant')
    plt.imshow(x_train[i].astype('uint8'), interpolation='nearest')
plt.show()

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot(figsize=(8, 5))
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

i = 0
prop_class = []
mis_class = []

# Convert predictions to binary if not already
Y_pred_tta_binary = (Y_pred_tta > 0.5).astype("int32")

# Collect 8 correctly classified
for i in range(len(Y_test)):
    if Y_test[i] == Y_pred_tta_binary[i]:
        prop_class.append(i)
    if len(prop_class) == 8:
        break

# Collect 8 misclassified
for i in range(len(Y_test)):
    if Y_test[i] != Y_pred_tta_binary[i]:
        mis_class.append(i)
    if len(mis_class) == 8:
        break

# Plot correctly classified
w = 60
h = 40
fig = plt.figure(figsize=(18, 10))
columns = 4
rows = 2

def Transfername(namecode):
    return "Benign" if namecode == 0 else "Malignant"

for i in range(len(prop_class)):
    ax = fig.add_subplot(rows, columns, i + 1)
    pred = Y_pred_tta_binary[prop_class[i]]
    actual = Y_test[prop_class[i]]
    ax.set_title("Predicted: " + Transfername(pred) + "\nActual: " + Transfername(actual))
    plt.imshow(X_test[prop_class[i]].astype('uint8'), interpolation='nearest')
    plt.axis('off')

plt.suptitle("Correctly Classified Examples")
plt.tight_layout()
plt.show()


# === 🔧 Misc ===
!pip install opencv-python-headless

!pip install tensorflow



benign_train_label = np.zeros(len(benign_train))
malign_train_label = np.ones(len(malign_train))
benign_test_label = np.zeros(len(benign_test))
malign_test_label = np.ones(len(malign_test))

X_train = np.concatenate((benign_train, malign_train), axis = 0)
Y_train = np.concatenate((benign_train_label, malign_train_label), axis = 0)
X_test = np.concatenate((benign_test, malign_test), axis = 0)
Y_test = np.concatenate((benign_test_label, malign_test_label), axis = 0)

s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]

s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
Y_test = Y_test[s]

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

# Clear previous models and free memory
K.clear_session()
gc.collect()

# Load ResNet50 base model
resnet = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Build your model using the base
model = build_model(resnet, lr=1e-4)

# Show model architecture
model.summary()

with open('history.json', 'w') as f:
    json.dump(history.history, f)

model.load_weights("weights.best.weights.h5")

Y_val_pred = (model.predict(x_val) > 0.5).astype("int32")

Y_pred = (model.predict(X_test) > 0.5).astype("int32")

tta_steps = 10
predictions = []

for i in tqdm(range(tta_steps)):
    preds = model.predict(
        train_generator.flow(X_test, batch_size=BATCH_SIZE, shuffle=False),
        steps=len(X_test) // BATCH_SIZE,
        verbose=0
    )
    predictions.append(preds)
    gc.collect()

Y_pred_tta = np.mean(predictions, axis=0)



