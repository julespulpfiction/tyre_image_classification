# build a cnn for binary classification of good and defective tyres

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split

s = 128

folder_path = '/content/drive/MyDrive/images'
defectives = os.listdir(os.path.join(folder_path, 'defective'))
goods = os.listdir(os.path.join(folder_path, 'good'))

# resize all the images to s x s, normalize and enlist the images
images = []
for im in defectives:
    img = plt.imread(os.path.join(folder_path, 'defective', im))
    img = np.resize(img, (s, s, 3))
    img = img / 255
    images.append(img)

for im in goods:
    img = plt.imread(os.path.join(folder_path, 'good', im))
    img = np.resize(img, (s, s, 3))
    img = img / 255
    images.append(img)

# create a dataframe with the images and their labels and shuffle it
df = pd.DataFrame({
    'filename': images,
    'label': ['defective'] * len(defectives) + ['good'] * len(goods)
}).sample(frac=1).reset_index(drop=True)

# split the dataframe into training and testing dataframes
train_df, test_df = train_test_split(df, test_size=0.1)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_labels = train_df['label'].replace({'defective': 0, 'good': 1})
test_labels = test_df['label'].replace({'defective': 0, 'good': 1})

train_labels.shape, test_labels.shape

train_images = np.array(train_df['filename'].tolist()).reshape(1670, s, s, 3)
test_images = np.array(test_df['filename'].tolist()).reshape(186, s, s, 3)

print('datasets created')

def build_model():
    model = Sequential()
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.2))
    model.add(layers.RandomZoom(0.1))
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(s, s, 3),
                                                    kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    opt = RMSprop(clipnorm=1.0)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# perform cross validation
k = 5
num_val_samples = len(train_images) // k
all_mae_histories = []

for i in range(k):
    val_data = train_images[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_labels[i * num_val_samples: (i + 1) * num_val_samples]

    partial_X = np.concatenate([train_images[:i * num_val_samples],
                                train_images[(i+1) * num_val_samples:]], axis=0)
    partial_y = np.concatenate([train_labels[:i * num_val_samples],
                            train_labels[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    history = model.fit(partial_X, partial_y,
                        validation_data=(val_data, val_targets),
                        epochs=20, batch_size=128, verbose=0)
    mae_history = history.history["val_accuracy"]
    all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories])
                                                        for i in range(20)]

print('Cross validation accuracy: ', np.mean(average_mae_history))

history = model.fit(train_images, train_labels, epochs=20, batch_size=128)

print(model.evaluate(test_images, test_labels))
