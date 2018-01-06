import os
import time
from datetime import datetime

import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from keras.layers import Dense
from keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# optional -> for state observation and tracking
from keras.callbacks import ModelCheckpoint

# pylint: disable=line-too-long

def load_pima_data(current_filepath=None):
    print "..loading pima indians dataset"
    dataset = pd.DataFrame.from_dict([])
    if current_filepath is None:
        # download directly from website
        dataset = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data",
            header=None).values
    else:
        # import from local directory
        dataset = pd.read_csv(current_filepath, header=None).values
    return train_test_split(dataset[:, 0:8], dataset[:, 8], test_size=0.25, random_state=87)


def run_nn(x_train_set, x_test_set, y_train_set, y_test_set, n_neurons, n_epochs, seed=155,
           history=True, del_files=True, validation_split=0.0, early_stopping=None):
    np.random.seed(seed)

    print "..creating model and layers"
    nn_model = Sequential()  # create model
    nn_model.add(Dense(n_neurons, input_dim=x_train_set.shape[1], activation='relu')) # hidden layer
    nn_model.add(Dense(1, activation='sigmoid'))  # output layer
    nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_callbacks = []

    if early_stopping is not None:
        model_callbacks = [early_stopping]

    if history:
        print "..training with history"
        filepath = "logs/nn_weights_%dneurons-{epoch:02d}.hdf5" % n_neurons
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0,
                                     save_weights_only=True,
                                     save_best_only=False, mode='max')
        model_callbacks.append(checkpoint)
        output = nn_model.fit(x_train_set, y_train_set, epochs=n_epochs, verbose=0,
                              batch_size=x_train_set.shape[0], callbacks=model_callbacks,
                              initial_epoch=0, validation_split=validation_split).history
        
        time.sleep(0.1)  # hack so that files can be opened in subsequent code
        
        temp_val_model = Sequential()  # create model
        temp_val_model.add(Dense(n_neurons, input_dim=x_train_set.shape[1], activation='relu'))  # hidden layer
        temp_val_model.add(Dense(1, activation='sigmoid'))  # output layer
        temp_val_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        test_over_time = []
        for i in range(len(output['loss'])):
            adj_i = i + 1
            current_filepath = "logs/nn_weights_%dneurons-%02d.hdf5" % (n_neurons, adj_i)
            temp_val_model.load_weights(current_filepath)
            scores = temp_val_model.evaluate(x_test_set, y_test_set, verbose=0)
            test_over_time.append(scores)
            # delete files once we're done with them
            if del_files:
                os.remove(current_filepath)
        test_over_time = np.array(test_over_time)
        output['test_loss'] = [row[0] for row in test_over_time]
        output['test_acc'] = [row[1] for row in test_over_time]
    else:
        print "..training without histroy"
        model_output = nn_model.fit(x_train_set, y_train_set, epochs=n_epochs, verbose=0,
                                    batch_size=x_train_set.shape[0], initial_epoch=0,
                                    callbacks=model_callbacks, validation_split=validation_split)
        validation_size = 0
        output = {}
        if validation_split > 0:
            validation_scores = nn_model.evaluate(model_output.validation_data[0],
                                                  model_output.validation_data[1],
                                                  verbose=0)
            validation_size = model_output.validation_data[0].shape[0]
            output['validation_loss'] = validation_scores[0]
            output['validation_acc'] = validation_scores[1]
        training_size = x_train_set.shape[0] - validation_size
        train_scores = nn_model.evaluate(x_train_set[0:training_size],
                                         y_train_set[0:training_size], verbose=0)
        test_scores = nn_model.evaluate(x_test_set, y_test_set, verbose=0)
        output['train_loss'] = train_scores[0]
        output['train_acc'] = train_scores[1]
        output['test_loss'] = test_scores[0]
        output['test_acc'] = test_scores[1]
    return output

def plot_this(neural_network_history, save_to_file_name):
    print "..plotting"
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(range(len(neural_network_history['loss'])), neural_network_history['loss'], color='blue', label='Training Set', lw=1)
    ax1.plot(range(len(neural_network_history['test_loss'])), neural_network_history['test_loss'], color='green', label='Test Set', lw=1)
    leg = ax1.legend(bbox_to_anchor=(0.7, 0.9), loc=2, borderaxespad=0., fontsize=10)
    ax1.set_xticklabels('')
    ax1.set_ylabel('Loss', fontsize=10)

    ax2.plot(range(len(neural_network_history['acc'])), neural_network_history['acc'], color='blue', label='Training Set', lw=1)
    ax2.plot(range(len(neural_network_history['test_acc'])), neural_network_history['test_acc'], color='green', label='Test Set', lw=1)
    ax2.set_xlabel('# Epochs', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_to_file_name, dpi=200)

def plot_this_2(neural_network_output_1, neural_network_output_2, save_to_file_name):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(range(len(neural_network_output_1['loss'])), neural_network_output_1['loss'], linestyle='-', color='blue',label='Training', lw=1)
    ax1.plot(range(len(neural_network_output_1['test_loss'])), neural_network_output_1['test_loss'], linestyle='-', color='green',label='Test', lw=1)
    ax1.plot(range(len(neural_network_output_2['loss'])), neural_network_output_2['loss'], linestyle='-', color='deepskyblue',label='Training (Standardised)', lw=1)
    ax1.plot(range(len(neural_network_output_2['test_loss'])), neural_network_output_2['test_loss'], linestyle='-', color='lightgreen',label='Test (Standardised)', lw=1)
    ax2.plot(range(len(neural_network_output_1['acc'])), neural_network_output_1['acc'], linestyle='-', color='blue',label='Training', lw=1)
    ax2.plot(range(len(neural_network_output_1['test_acc'])), neural_network_output_1['test_acc'], linestyle='-', color='green',label='Test', lw=1)
    ax2.plot(range(len(neural_network_output_2['acc'])), neural_network_output_2['acc'], linestyle='-', color='deepskyblue',label='Training (Standardised)', lw=1)
    ax2.plot(range(len(neural_network_output_2['test_acc'])), neural_network_output_2['test_acc'], linestyle='-', color='lightgreen',label='Test (Standardised)', lw=1)
    leg = ax1.legend(bbox_to_anchor=(0.6, 0.9), loc=2, borderaxespad=0.,fontsize=10)
    ax1.set_xticklabels('')
    ax2.set_xlabel('# Epochs',fontsize=10)
    ax1.set_ylabel('Loss',fontsize=10)
    ax2.set_ylabel('Accuracy',fontsize=10)
    ax2.annotate('Overfitting starts', xy=(80, 0.8), xytext=(100, 0.5), fontsize=10, arrowprops=dict(facecolor='black', shrink=0.0,headwidth=10))
    plt.tight_layout()
    plt.savefig(save_to_file_name, dpi=200)

def plot_this_3(nn_output_unscaled, nn_output_scaled, save_to_file_name):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(range(len(nn_output_unscaled['loss'])), nn_output_unscaled['loss'], linestyle='-', color='blue',label='Training', lw=1)
    ax1.plot(range(len(nn_output_unscaled['val_loss'])), nn_output_unscaled['val_loss'], linestyle='-', color='purple',label='Validation', lw=1)
    ax1.plot(range(len(nn_output_unscaled['test_loss'])), nn_output_unscaled['test_loss'],linestyle='-', color='green',label='Test', lw=1)
    ax1.plot(range(len(nn_output_scaled['loss'])), nn_output_scaled['loss'], linestyle='-', color='deepskyblue',label='Training (Standardised)', lw=1)
    ax1.plot(range(len(nn_output_scaled['val_loss'])), nn_output_scaled['val_loss'], linestyle='-', color='mediumpurple',label='Validation (Standardised)', lw=1)
    ax1.plot(range(len(nn_output_scaled['test_loss'])), nn_output_scaled['test_loss'], linestyle='-', color='lightgreen',label='Test (Standardised)', lw=1)
    ax2.plot(range(len(nn_output_unscaled['acc'])), nn_output_unscaled['acc'], linestyle='-', color='blue',label='Training', lw=1)
    ax2.plot(range(len(nn_output_unscaled['val_acc'])), nn_output_unscaled['val_acc'], linestyle='-', color='purple',label='Validation', lw=1)
    ax2.plot(range(len(nn_output_unscaled['test_acc'])), nn_output_unscaled['test_acc'],linestyle='-', color='green',label='Test', lw=1)
    ax2.plot(range(len(nn_output_scaled['acc'])), nn_output_scaled['acc'], linestyle='-', color='deepskyblue',label='Training (Standardised)', lw=1)
    ax2.plot(range(len(nn_output_scaled['val_acc'])), nn_output_scaled['val_acc'], linestyle='-', color='mediumpurple',label='Validation (Standardised)', lw=1)
    ax2.plot(range(len(nn_output_scaled['test_acc'])), nn_output_scaled['test_acc'], linestyle='-', color='lightgreen',label='Test (Standardised)', lw=1)
    leg = ax1.legend(bbox_to_anchor=(0.5, 0.95), loc=2, borderaxespad=0.,fontsize=10)
    ax1.set_xticklabels('')
    ax2.set_xlabel('# Epochs',fontsize=10)
    ax1.set_ylabel('Loss',fontsize=10)
    ax2.set_ylabel('Accuracy',fontsize=10)
    plt.tight_layout()
    plt.savefig(save_to_file_name, dpi=200)

def main():
    datestring = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # load data
    x_train, x_test, y_train, y_test = load_pima_data()
    
    # initial run
    nn_output = run_nn(x_train, x_test, y_train, y_test, 1000, 1000)
    plot_this(nn_output, "logs/%s_plot_1 _plain.png" % datestring)

    # run with scaled data
    scaler = StandardScaler()
    nn_output_scaled = run_nn(scaler.fit_transform(x_train),
                              scaler.fit_transform(x_test),
                              y_train, y_test, 1000, 1000)
    plot_this_2(nn_output, nn_output_scaled, "logs/%s_plot_2_scaled.png" % datestring)

    # run with scaled data and stop criteria
    early_stop_crit = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
    nn_output_stop = run_nn(x_train, x_test, y_train, y_test, 1000, 1000,
                            validation_split=0.2, early_stopping=early_stop_crit)
    nn_output_stop_scaled = run_nn(scaler.fit_transform(x_train), scaler.fit_transform(x_test),
                                    y_train, y_test, 1000, 1000,validation_split=0.2, early_stopping=early_stop_crit)
    plot_this_3(nn_output_stop, nn_output_stop_scaled, "logs/%s_plot_3_scaledstop.png" % datestring)


if __name__ == '__main__':
    print "\nWelcome!"
    main()
