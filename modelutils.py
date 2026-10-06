"""
modelutils.py -- the forecasting models and the helpers used to train and evaluate them.

Contents
    FeedBack                  Autoregressive (closed-loop) 2-layer LSTM. It warms up on an input window, then
                              predicts one step at a time, feeding each prediction (plus the constant DBS
                              inputs) back in as the next input.
    DirectLSTM                The same 2-layer LSTM with a Dense head that predicts all output steps at once.
    ModelUtils.plot_each_column           Save one plot per predicted column for a single test example.
    ModelUtils.compile_and_fit            Compile with MAE loss + Adam and fit on (X, y) numpy arrays.
    ModelUtils.prepare_time_series_data   Slide a window over every gait cycle to build (inputs, labels) for train/val.
    ModelUtils.prepare_test_series_data   Take one window from the start of every gait cycle for testing.
    ModelUtils.plot_channel               Plot inputs, labels and predictions for a few test examples.
    SmoothingSignal           Causal simple moving average and median filter (not used by the main scripts).

Used by train.py, dataset.py and finetune.py.
"""
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import scipy
import scipy.io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pickle
from utils import LoadData, NormalizedData
from configs import path_to_data, path_to_save_models_results, path_to_save_models_images

class FeedBack(tf.keras.Model):
  """
  Closed-loop (autoregressive) 2-layer LSTM.

  Args:
      lstm_units      number of units in each of the two LSTM layers
      dense_units     size of the hidden Dense layer before the output layer
      out_steps       number of future timesteps to predict
      num_inputs      number of input features per timestep (constants + predicted features)
      num_features    number of predicted features (outputs per timestep)
      num_constants   number of leading input columns that stay constant and are re-attached to
                      every fed-back prediction (the 4 DBS settings)
      activation_lstm activation of the LSTM cells
      activation_dense currently unused (the Dense layers are linear)
      use_bias        currently unused (both Dense layers are built with use_bias=False)
      l2_reg          L2 weight on the LSTM kernels

  Input shape  (batch, input_width, num_inputs)
  Output shape (batch, out_steps, num_features)
  """
  def __init__(self, lstm_units, dense_units, out_steps, num_inputs, num_features, num_constants,
  activation_lstm='tanh', activation_dense='linear',use_bias = True, l2_reg = 0.01):
    super().__init__()
    self.out_steps = out_steps
    self.lstm_units = lstm_units
    self.dense_units = dense_units
    self.num_inputs = num_inputs
    self.num_features = num_features
    self.num_constants = num_constants
    self.lstm_cell1 = tf.keras.layers.LSTMCell(
        lstm_units,
        activation = activation_lstm,
        kernel_regularizer = tf.keras.regularizers.L2(l2_reg)
    )
    self.lstm_cell2 = tf.keras.layers.LSTMCell(
        lstm_units,
        activation = activation_lstm,
        kernel_regularizer = tf.keras.regularizers.L2(l2_reg)
    )
    self.lstm_rnn1 = tf.keras.layers.RNN(self.lstm_cell1, return_state=True, return_sequences = True)
    self.lstm_rnn2 = tf.keras.layers.RNN(self.lstm_cell2, return_state = True)
    # self.lstm1 = tf.keras.layers.LSTM(units, return_sequences = True, return_state = True)
    # self.lstm2 = tf.keras.layers.LSTM(units, return_state = True)
    # self.densor1 = tf.keras.layers.Dense(32)
    self.densor1 = tf.keras.layers.Dense(
        dense_units,
        # activation = activation_dense,
        use_bias = False,
        # kernel_regularizer = tf.keras.regularizers.L2(l2_reg)
    )
    self.densor2 = tf.keras.layers.Dense(
        num_features,
        use_bias = False
    )

def warmup(self, inputs):
    """Run the whole input window through both LSTM layers; return the first prediction and both LSTM states."""
    x, *state1 = self.lstm_rnn1(inputs)
    x, *state2 = self.lstm_rnn2(x)
    pre_pred = self.densor1(x)
    prediction = self.densor2(pre_pred)
    return prediction, state1, state2

FeedBack.warmup = warmup

def call(self, inputs, training=None):
    """
    Predict out_steps timesteps. After warmup, each step concatenates the constant inputs
    (taken from the last input timestep) with the previous prediction and advances both LSTM cells by one step.
    """
    predictions = []
    # inputs.shape => (batch, time, features)
    prediction, state1, state2 = self.warmup(inputs)
    # Insert the first prediction.
    predictions.append(prediction)
    # Run the rest of the prediction steps.
    dbs_inputs = inputs[:, -1, :self.num_constants]
    for n in range(1, self.out_steps):
    # Use the last prediction as input.
        x = prediction
        x = tf.concat([dbs_inputs, x ], axis = -1)
        # Execute one lstm step.
        x, state1 = self.lstm_cell1(x, states=state1)
        x, state2 = self.lstm_cell2(x, states = state2)
        # Convert the lstm output to a prediction.
        pre_pred = self.densor1(x)
        prediction = self.densor2(pre_pred)
        # Add the prediction to the output.
        predictions.append(prediction)

    # predictions.shape => (time, batch, features)
    predictions = tf.stack(predictions)
    # predictions.shape => (batch, time, features)
    predictions = tf.transpose(predictions, [1, 0, 2])
    return predictions

FeedBack.call = call

class DirectLSTM(tf.keras.Model):
  """
  Direct (single-shot) multi-step model: the alternative to FeedBack's autoregressive loop.

  The same 2-layer LSTM reads the input window; a Dense head then predicts all out_steps
  timesteps at once from the final LSTM output. There is no feedback loop, so the LSTMs run only
  input_width steps (instead of input_width + out_steps - 1) and errors cannot accumulate step by step.
  The constant DBS inputs reach the head only through the LSTM state; FeedBack instead re-feeds them
  at every output step, held fixed over the whole horizon.

  Args:
      lstm_units      number of units in each of the two LSTM layers
      dense_units     size of the hidden Dense layer before the output layer
      out_steps       number of future timesteps to predict
      num_features    number of predicted features (outputs per timestep)
      l2_reg          L2 weight on the LSTM kernels

  Input shape  (batch, input_width, num_inputs)
  Output shape (batch, out_steps, num_features)  -- same as FeedBack
  """
  def __init__(self, lstm_units, dense_units, out_steps, num_features, l2_reg = 0.01):
    super().__init__()
    self.out_steps = out_steps
    self.num_features = num_features
    self.lstm1 = tf.keras.layers.LSTM(lstm_units, return_sequences = True,
                                      kernel_regularizer = tf.keras.regularizers.L2(l2_reg))
    self.lstm2 = tf.keras.layers.LSTM(lstm_units,
                                      kernel_regularizer = tf.keras.regularizers.L2(l2_reg))
    self.densor1 = tf.keras.layers.Dense(dense_units, use_bias = False)
    self.densor2 = tf.keras.layers.Dense(out_steps * num_features, use_bias = False)
    self.reshape = tf.keras.layers.Reshape((out_steps, num_features))

  def call(self, inputs, training=None):
    x = self.lstm1(inputs)
    x = self.lstm2(x)
    x = self.densor1(x)
    x = self.densor2(x)
    return self.reshape(x)

class ModelUtils:
    def compile_and_fit(model, train_data, val_data, params, patience = 20, batch_size = 64, shuffle = False,
                        use_early_stopping = False, extra_callbacks = None):
        """
        Compile with MAE loss and Adam, then fit. Defaults: batch_size 64, no shuffling, no early stopping.
        model - neural network model
        data - a tuple contain X and y
        params: a tuple contains hyperparameters (learning rate, epochs)
        use_early_stopping: stop when val_loss has not improved for `patience` epochs and restore the best weights
        extra_callbacks: list of additional Keras callbacks (e.g. checkpointing, logging)
        return model after compile and fitting
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        lr, MAX_EPOCHS = params
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                        patience = patience,
                                                        mode = 'min',
                                                        restore_best_weights = True)
        callbacks = [early_stopping] if use_early_stopping else []
        callbacks += extra_callbacks or []
        model.compile(loss = tf.keras.losses.MeanAbsoluteError(),
                    optimizer = tf.keras.optimizers.Adam(learning_rate = lr),
                    metrics = [tf.keras.metrics.MeanAbsoluteError()])
        history = model.fit(X_train,
                        y_train,
                        epochs = MAX_EPOCHS,
                        validation_data = (X_val, y_val),
                        batch_size = batch_size,
                        shuffle = shuffle,
                        # validation_split = 0.2,
                        callbacks = callbacks
                        )
        return history



def prepare_time_series_data(series_list, input_width, label_width, shift, numinputs, numoutputs, stride = 1):
    """
    Build training windows from every gait cycle long enough to hold one.

    series_list  list of 2D arrays (timesteps, features), one per gait cycle
    stride       number of timesteps between the starts of consecutive windows (1 = every timestep)
    returns      inputs (n, input_width, numinputs), labels (n, label_width, numoutputs);
                 labels are the last numoutputs columns
    """
    inputs_all = []
    labels_all = []
    
    for series in series_list:
        total_window_size = input_width + shift
        s = series
        # Check if the current series is long enough
        if s.shape[0] >= total_window_size:
            # Split the time series into windows of input and label
            for start in range(0, s.shape[0] - total_window_size + 1, stride):
                window = s[start:start + total_window_size]
                inputs = window[:input_width, :numinputs]
                labels = window[-label_width:, numinputs-numoutputs:numinputs]
                inputs_all.append(inputs)
                labels_all.append(labels)
    
    # Convert list to numpy arrays
    inputs_all = np.array(inputs_all)
    labels_all = np.array(labels_all)
    
    return inputs_all, labels_all
ModelUtils.prepare_time_series_data = prepare_time_series_data

def prepare_test_series_data(test_list, input_width, label_width, shift, numinputs, numoutputs):
    """Like prepare_time_series_data, but takes only one window from the start of each gait cycle."""
    test_input, test_label = [], []
    for series in test_list:
        total_window_size = input_width + shift
        if series.shape[0] >= total_window_size:
            window = series[:total_window_size]
            inputs = window[:input_width, :numinputs]
            labels = window[-label_width:, numinputs-numoutputs:numinputs]
            test_input.append(inputs)
            test_label.append(labels)
    return np.array(test_input), np.array(test_label)

ModelUtils.prepare_test_series_data = prepare_test_series_data

def plot_channel(inputs, labels, predictions, name, plot_col_index = 5, plot_col ='channel1', max_subplots = 3):
    """
    Plot input history, true labels and predictions for max_subplots examples, and save the figure
    to configs.path_to_save_models_images/<name>.png. plot_col_index is the input column to plot;
    the matching label/prediction column is plot_col_index - 5.
    """
    plt.figure(figsize = (12, 8))
    print(f"debug: {predictions.shape}")
    # Find the column index
    for n in range(max_subplots):
        # retrieve the data
        input_data, label_data = inputs[n:n+1, :, :], labels[n: n+1, :, :]
        input_indices = [i for i in range(input_data.shape[1])]
        label_indices = [i for i in range(input_data.shape[1], input_data.shape[1] + label_data.shape[1])]
        label_col_index = plot_col_index - 5
        # plot inputs (only the first example in each batch)
        plt.subplot(max_subplots, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        print(f"Debug:\nInput shape: {input_data[0, :, plot_col_index].shape}\n")
        plt.plot(input_indices, input_data[0, :, plot_col_index],
             label=f'Inputs_{input_data[0, 0, :5]}', marker='.', markersize=10, zorder=-10)

        plt.scatter(label_indices, label_data[:,:, label_col_index],
                label='Labels', c='k', s=10)

        plt.scatter(label_indices, predictions[n:n+1,:, label_col_index],
            marker='X', label='Predictions',
            c='#ff7f0e', s=10)
        plt.legend()

    plt.xlabel(plot_col)
    if name is not None:
        plt.savefig(os.path.join(path_to_save_models_images,f"{name}.png"), dpi=300)
    plt.show()
    plt.clf()
ModelUtils.plot_channel = plot_channel

def plot_each_column(inputs, labels, predictions, output_names, num_constants, name_prefix, example_index = 0):
    """
    Save one image per predicted column for a single test example: the input history, true labels and
    predictions of that column. Files go to configs.path_to_save_models_images/<name_prefix>_<column>.png.

    inputs        (m, input_width, num_inputs); the predicted columns follow the num_constants constant columns
    labels        (m, out_steps, num_outputs)
    predictions   (m, out_steps, num_outputs)
    output_names  names of the predicted columns, in the order of the last axis of labels/predictions
    example_index which test example to plot
    """
    input_width = inputs.shape[1]
    input_indices = range(input_width)
    label_indices = range(input_width, input_width + labels.shape[1])
    for j, col_name in enumerate(output_names):
        col_name = col_name.strip()
        plt.figure(figsize = (12, 4))
        plt.plot(input_indices, inputs[example_index, :, num_constants + j],
                 label = 'Inputs', marker = '.', markersize = 10, zorder = -10)
        plt.scatter(label_indices, labels[example_index, :, j], label = 'Labels', c = 'k', s = 10)
        plt.scatter(label_indices, predictions[example_index, :, j], marker = 'X', label = 'Predictions',
                    c = '#ff7f0e', s = 10)
        plt.title(f"{col_name} (test example {example_index})")
        plt.ylabel(f'{col_name} [normed]')
        plt.xlabel('Timestep')
        plt.legend()
        plt.savefig(os.path.join(path_to_save_models_images, f"{name_prefix}_{col_name}.png"), dpi = 300)
        plt.close()
ModelUtils.plot_each_column = plot_each_column

from scipy.signal import medfilt
from scipy import signal
class SmoothingSignal:
    """Causal smoothing filters over a 1D series: simple moving average and median filter."""
    def __init__(self, kernel_size_sma, kernel_size_median_filter):
        self.kernel_size_sma = kernel_size_sma
        self.kernel_size_median_filter = kernel_size_median_filter
    
    def simple_moving_average(self, time_series_data):
        sma = []
        for i in range(len(time_series_data)):
            start = max(0, i - self.kernel_size_sma +  1)
            end = i + 1
            window = time_series_data[start:end]
            sma.append(np.mean(window))
        return sma

    def find_median(self, window):
        if window.shape[0] == 1:
            return window[0]
        else:
            sorted_window=np.sort(window)
            if sorted_window.shape[0]%2 == 0:
                start = sorted_window.shape[0]//2 - 1
                end = sorted_window.shape[0]//2 + 1
                return np.mean(sorted_window[start:end])
            else:
                return sorted_window[sorted_window.shape[0]//2]

    
    def median_filter(self, time_series_data):
        medfilter = []
        for i in range(len(time_series_data)):
            start = max(0, i - self.kernel_size_median_filter + 1)
            end = i + 1
            window = time_series_data[start:end]
            medfilter.append(self.find_median(window))
        return np.array(medfilter)



        




