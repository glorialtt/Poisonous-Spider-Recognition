from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D
from keras.layers import add, Flatten
from keras.utils import plot_model
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
from keras import backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.callbacks import ModelCheckpoint
import h5py
import yaml


# loading meta
def loading_meta(filepath):
    meta = {}

    with h5py.File(filepath, 'r') as f:
        meta_group = f['meta']
        meta['training_args'] = yaml.load(
            meta_group.attrs['training_args']
        )
        for k in meta_group.keys():
            meta[k] = list(meta_group[k])
        return meta


def get_last_status(model):
    last_epoch = -1
    last_meta = {}
    if os.path.exists('/content/mjcdrive/My Drive/downloads/result/res50.h5'):
        model.load_weights('/content/mjcdrive/My Drive/downloads/result/res50.h5')
        last_meta = loading_meta('/content/mjcdrive/My Drive/downloads/result/res50.h5')
        last_epoch = last_meta.get('epochs')[-1]
    return last_epoch, last_meta


# confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')


# parameters
batch_size = 32
train_path = "/content/spider-recognition/train"
test_path = "/content/spider-recognition/test"
validation_path = "/content/spider-recognition/validation"
epoch = 60
CLASS = 10

# preparing data
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=batch_size,
                                                    shuffle=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=batch_size,
                                                  shuffle=False)

vali_datagen = ImageDataGenerator(rescale=1. / 255)
vali_generator = vali_datagen.flow_from_directory(validation_path, target_size=(224, 224), batch_size=batch_size)

if os.path.exists('/content/mjcdrive/My Drive/downloads/result/adam/res50.h5'):
    model = load_model('/content/mjcdrive/My Drive/downloads/result/adam/res50.h5')

# lastpoch, lastmeta = get_last_status(model)
# print(lastmeta)

loss, acc, top_acc = model.evaluate_generator(test_generator, steps=test_generator.n / batch_size)
print('Test result:loss:%f,acc:%f,top_acc:%f' % (loss, acc, top_acc))

print('start prediction')
predictions = model.predict_generator(test_generator, steps=test_generator.n / batch_size)
predict_label = np.argmax(predictions, axis=1)

true_label = test_generator.classes
print('Confusion Matrix')
con_matrix = confusion_matrix(true_label, predict_label)
plot_confusion_matrix(con_matrix, classes=range(CLASS))