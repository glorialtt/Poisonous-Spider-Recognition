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
from keras.optimizers import SGD


# K.set_image_dim_ordering('th')


# parameters
batch_size = 64
train_path = "/content/spider-recognition/train"
test_path = "/content/spider-recognition/test"
validation_path = "/content/spider-recognition/validation"
epoch = 60
CLASS = 10


# Build the model
def Conv_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='SAME', name=None):
    if name is not None:
        batch_name = name + 'bn'
        conv_name = name + 'conv'
    else:
        batch_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding='SAME', strides=strides, name=conv_name)(x)
    x = BatchNormalization(axis=3, name=batch_name)(x)
    return x


def bottleNeck(inpt, nb_filters, strides=(1, 1), with_shortcut=False):
    filter1 = nb_filters[0]
    filter2 = nb_filters[1]
    filter3 = nb_filters[2]
    x = Conv_BN(inpt, nb_filter=filter1, kernel_size=1, strides=strides, padding='SAME')
    x = Activation('relu')(x)
    x = Conv_BN(x, nb_filter=filter2, kernel_size=3, padding='SAME')
    x = Activation('relu')(x)
    x = Conv_BN(x, nb_filter=filter3, kernel_size=1, padding='SAME')
    if with_shortcut:
        shortcut = Conv_BN(inpt, nb_filter=filter3, strides=strides, kernel_size=1)
        x = add([x, shortcut])
    else:
        x = add([x, inpt])
    x = Activation('relu')(x)
    return x


def Res50(width=224, height=224, channel=3, classes=10):
    inpt = Input(shape=(224, 224, 3))
    x = ZeroPadding2D((3, 3))(inpt)
    x = Conv_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='VALID')
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='SAME')(x)

    # The second Conv
    x = bottleNeck(x, nb_filters=[64, 64, 256], strides=(1, 1), with_shortcut=True)
    x = bottleNeck(x, nb_filters=[64, 64, 256])
    x = bottleNeck(x, nb_filters=[64, 64, 256])

    # The thrid Conv
    x = bottleNeck(x, nb_filters=[128, 128, 512], strides=(2, 2), with_shortcut=True)
    x = bottleNeck(x, nb_filters=[128, 128, 512])
    x = bottleNeck(x, nb_filters=[128, 128, 512])
    x = bottleNeck(x, nb_filters=[128, 128, 512])

    # The fourth Conv
    x = bottleNeck(x, nb_filters=[256, 256, 1024], strides=(2, 2), with_shortcut=True)
    x = bottleNeck(x, nb_filters=[256, 256, 1024])
    x = bottleNeck(x, nb_filters=[256, 256, 1024])
    x = bottleNeck(x, nb_filters=[256, 256, 1024])
    x = bottleNeck(x, nb_filters=[256, 256, 1024])
    x = bottleNeck(x, nb_filters=[256, 256, 1024])

    # The fifth Conv
    x = bottleNeck(x, nb_filters=[512, 512, 2048], strides=(2, 2), with_shortcut=True)
    x = bottleNeck(x, nb_filters=[512, 512, 2048])
    x = bottleNeck(x, nb_filters=[512, 512, 2048])

    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    return model


def create_Model():
    model = Res50(CLASS)
    model.summary()

    plot_model(model, to_file='/content/mjcdrive/My Drive/downloads/result/res50.png')
    sgd = SGD(lr=0.01, decay=1e-5, momentum=0, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc', top_k_categorical_accuracy])
    print("Model compiled")

    return model

#Define a meta checkpoint
class MetaChecker(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_acc', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, training_args=None, meta=None):

        super(MetaChecker, self).__init__(filepath, monitor=monitor,
                                          verbose=verbose,
                                          save_best_only=save_best_only,
                                          save_weights_only=save_weights_only,
                                          mode=mode,
                                          period=period)
        self.filepath = filepath
        self.new_file_override = True
        self.meta = meta or {'epochs': [], self.monitor: []}

        if training_args:
            self.meta['training_args'] = training_args

    def on_train_begin(self, logs={}):
        if self.save_best_only:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.best = max(self.meta[self.monitor], default=-np.inf)
            else:
                self.best = min(self.meta[self.monitor], default=np.inf)
        super(MetaChecker, self).on_train_begin(logs)

    def on_epoch_end(self, epoch, logs={}):
        if self.save_best_only:
            current = logs.get(self.monitor)
            if self.monitor_op(current, self.best):
                self.new_file_override = True
            else:
                self.new_file_override = False

        super(MetaChecker, self).on_epoch_end(epoch, logs)


        self.meta['epochs'].append(epoch)
        for k, v in logs.items():
            self.meta.setdefault(k, []).append(v)

        filepath = self.filepath.format(epoch=epoch, **logs)

        if self.new_file_override and self.epochs_since_last_save==0:
            with h5py.File(filepath, 'r+') as f:
                meta_group = f.create_group('meta')
                meta_group.attrs['training_args'] = yaml.dump(self.meta.get('training_args', '{}'))
                meta_group.create_dataset('epochs', data=np.array(self.meta['epochs']))
                for k in logs:
                    meta_group.create_dataset(k, data=np.array(self.meta[k]))





# generating some picture
def plot_training(self, history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b-')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(epochs, loss, 'b-')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()

# learning curves
def learning_curves(history):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(history.history['loss'], color='r', label='Training Loss')
    ax[0].plot(history.history['val_loss'], color='g', label='Validation Loss')
    ax[0].legend(loc='best', shadow=True)
    ax[0].grid(True)

    ax[1].plot(history.history['acc'], color='r', label='Training Accuracy')
    ax[1].plot(history.history['val_acc'], color='g', label='Validation Accuracy')
    ax[1].legend(loc='best', shadow=True)
    ax[1].grid(True)

# loading meta
def loading_meta(filepath):
    meta = {}

    with h5py.File(filepath, 'r') as f:
        meta_group=f['meta']
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



# preparing data
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=batch_size,
                                                    shuffle=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=batch_size)

vali_datagen = ImageDataGenerator(rescale=1. / 255)
vali_generator = vali_datagen.flow_from_directory(validation_path, target_size=(224, 224), batch_size=batch_size)




if os.path.exists('/content/mjcdrive/My Drive/downloads/result/res50.h5'):
    model = load_model('/content/mjcdrive/My Drive/downloads/result/res50.h5')
else:
    model = create_Model()



#if the model is trained for the first time, use this check point
check_point = MetaChecker('/content/mjcdrive/My Drive/downloads/result/res50.h5', monitor='val_acc',
                          save_best_only=True, save_weights_only=False,verbose=1)

# # if not the first time
# last_epoch, last_meta = get_last_status(model)
# check_point = MetaChecker('/content/mjcdrive/My Drive/downloads/result/res50.h5', monitor='val_acc',
#                           save_best_only=True, save_weights_only=False,verbose=1, meta = last_meta)
# print(last_meta)


# # learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 3, verbose = 1, factor=0.5, min_lr = 0.00001)

# #For the first time
history = model.fit_generator(train_generator, validation_data=vali_generator, epochs=epoch,
                              steps_per_epoch=train_generator.n / batch_size,
                              validation_steps=vali_generator.n / batch_size,
                              callbacks=[check_point])
model.save('/content/mjcdrive/My Drive/downloads/result/res50.h5', overwrite=True)
print('model saved')

# #If not the first time
# history = model.fit_generator(train_generator, validation_data=vali_generator, epochs=epoch,
#                               steps_per_epoch=train_generator.n / batch_size,
#                               validation_steps=vali_generator.n / batch_size,
#                               callbacks=[check_point],
#                               initial_epoch=last_epoch+1)


loss, acc, top_acc = model.evaluate_generator(test_generator, steps=test_generator.n / batch_size)
print('Test result:loss:%f,acc:%f,top_acc:%f' % (loss, acc, top_acc))
# history = model
# plot_training(history)
# learning_curves(model)


# plot confusion matrix
# print('start prediction')
# predictions = model.predict_generator(test_generator, steps=test_generator.n / batch_size)
# predict_label = np.argmax(predictions, axis = 1)
# true_label = test_generator.classes
# # pd.crosstab(true_label,predict_label,rownames=['label'],colnames=['predict'])

# con_matrix = confusion_matrix(true_label, predict_label)
# plot_confusion_matrix(con_matrix, classes=range(CLASS)