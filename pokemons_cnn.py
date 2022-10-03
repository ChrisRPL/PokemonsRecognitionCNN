import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os, shutil
from os.path import exists

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense
from keras import metrics

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

def prepare_data(data_folder, data_description):
    data_df = pd.read_csv(data_description)
    pokemon_types = data_df['Type1'].unique()

    test_dir_exists = os.path.exists('test')
    train_dir_exists = os.path.exists('train')

    if not test_dir_exists:
        os.makedirs('test')

    if not train_dir_exists:
        os.makedirs('train')
    
    for type in pokemon_types:
        type_test_dir_path = f'test/{type}'
        type_train_dir_path = f'train/{type}'

        type_dir_in_test_exists = os.path.exists(type_test_dir_path)
        type_dir_in_train_exists = os.path.exists(type_train_dir_path)

        if not type_dir_in_test_exists:
            os.makedirs(type_test_dir_path)

        if not type_dir_in_train_exists:
            os.makedirs(type_train_dir_path)
    
    #we generate folders only once
    if not test_dir_exists and not train_dir_exists:
        for index, row in data_df.iterrows():
            name = row['Name']
            type = row['Type1']

            try:
                #we get 20% for test data
                if index % 5 == 0:
                    shutil.copy2(f'{data_folder}/{name}.jpg', f'test/{type}/{name}.jpg')
                else:
                    shutil.copy2(f'{data_folder}/{name}.jpg', f'train/{type}/{name}.jpg')
            except:
                if index % 5 == 0:
                    shutil.copy2(f'{data_folder}/{name}.png', f'test/{type}/{name}.png')
                else:
                    shutil.copy2(f'{data_folder}/{name}.png', f'train/{type}/{name}.png')
    
    image_gen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, rescale=1/255, shear_range=0.2, zoom_range=0.2,
                                    horizontal_flip=True, fill_mode='nearest')
    
    input_shape = (64,64,3)
    batch_size = 16

    train_image_gen = image_gen.flow_from_directory('train', target_size=input_shape[:2], batch_size=batch_size, class_mode='categorical')
    test_image_gen = image_gen.flow_from_directory('test', target_size=input_shape[:2], batch_size=batch_size, class_mode='categorical')

    return train_image_gen, test_image_gen, pokemon_types

def init_model():
    np.random.seed(1000)
    model = Sequential()
    
    #ADD 4 CONVOLUTIONAL LAYERS WITH MAX POOLING
    model.add(Conv2D(filters=32, input_shape=(64,64,3), kernel_size=(3,3) , padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))

    model.add(Conv2D(filters=64, kernel_size=(3,3),  padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))

    # FLATTEN IMAGES AND PASS IT TO THE HIDDEN LAYER
    model.add(Flatten())

    #ADD 3 DENSE HIDDEN LAYERS WITH DROPOUT TO PREVENT OVERFITTING
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
  
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # OUR CLASSIFIER OUTPUT LAYER
    model.add(Dense(18))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-ds', '--data_description', default='pokemon.csv', help='A csv file with pokemons description.')
    parser.add_argument('-i', '--images_data_folder', default='pokemon', help='A folder with pokemons images.')

    return parser.parse_args()

def main(args):
    data_folder = args.images_data_folder;
    data_description = args.data_description

    train_image_gen, test_image_gen, labels_names = prepare_data(data_folder, data_description)

    if exists('model_pokemons.h5'):
        model = load_model('model_pokemons.h5')
        results = model.evaluate(train_image_gen)
        y_pred=model.predict(train_image_gen)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = np.sort(y_pred)
        cf_matrix = confusion_matrix(train_image_gen.classes, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=labels_names)
    
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig('cf_matrix.png')
        plt.show()
        
    else:
        model = init_model()

        results = model.fit_generator(train_image_gen, epochs=500, validation_data=test_image_gen, shuffle=True)

        hist_frame = pd.DataFrame(results.history)
        hist_frame.loc[:,['accuracy', 'val_accuracy']].plot()
        plt.show()

        y_pred=model.predict(test_image_gen)
        y_pred = np.argmax(y_pred, axis=1)
        cf_matrix = confusion_matrix(test_image_gen.classes, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=labels_names)

        disp.plot(cmap=plt.cm.Blues)
        plt.savefig('cf_matrix.png')
        plt.show()

        model.save('model_pokemons.h5')

if __name__ == '__main__':
    main(parse_arguments())