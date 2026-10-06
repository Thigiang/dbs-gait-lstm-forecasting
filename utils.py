"""
utils.py -- data loading, flattening/normalization and plotting helpers.

Classes
    LoadData        Reads the session .mat files in a folder. Each file holds an (m, 1) object array;
                    every element is one gait cycle of shape (timesteps, 45).
    NormalizedData  Flattens a list of variable-length gait cycles into one 2D array, normalizes it
                    ('std', 'mima', 'robust', 'yeo' via scikit-learn, or 'mean_normalization'),
                    and splits it back into per-cycle arrays.
    Visualizer      Small plotting helpers (DBS settings, band power traces, prediction vs. label).

The 45 columns of each gait cycle are:
    4 DBS parameters (DBSLAmp, DBSLFreq, DBSRAmp, DBSRFreq)
    40 band powers  (Left/Right x Channel 1-4 x Delta/Theta/Alpha/Beta/Gamma)
    WPI

The commented-out WindowGenerator block is an older tf.data-based windowing approach kept for reference.
"""
import os
import pandas as pd
import numpy as np
import scipy
import scipy.io
import math
import matplotlib.pyplot as plt
from configs import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
import tensorflow as tf
import itertools



class LoadData:
    def __init__(self, folder_path):
        self.folder_path = folder_path
    def get_files_name(self):
        """
        if there is a .DS_Store in the folder directory if we use os.listdir(folder_path),
        we will remove this folder and obtain a list of files' name that has data in it.
        """
        all_files_in_folder_path = os.listdir(self.folder_path)
        # remove .DS_Store
        file_names = []
        for name in all_files_in_folder_path:
            if name[-4:] == ".mat":
                file_names.append(name)
        return file_names

    def load_single_data(self, single_file_path):
        """
        inputs -- single_file_path: a string which is the directory to a single data file
        return: -- df: a numpy array of shape (n_i, 1) where n_i = number of sessions in the file. 
                       Each sessions has shape (Txi, 45) where Txi is the number of timesteps in each session. Tx is different for different session
        """
        try:
            data_orig = scipy.io.loadmat(single_file_path, variable_names = None)
            # print(f"keys: {data_orig.keys()}")
            variable_name = list(data_orig.keys())[-1] 
            df = data_orig[variable_name]
            return np.array(df, dtype = 'object')
        except ValueError as e:
            print(f"Error loading {single_file_path}: {e}")
            return None

    def load_data_by_index(self, idx = 0):
        """
        load idx th dataset
        return df: 2D numpy array of shape (m, 1) where m is the number of examples.
                   Each example is a timecity with shape (t, 45) where t is the number of timesteps and 45 represents 45 columns
        """
        file_names = self.get_files_name()
        # print(f"file names: {file_names}")
        single_file_path = os.path.join(self.folder_path, file_names[idx])
        df = self.load_single_data(single_file_path)
        return df





# class WindowGenerator():
#     def __init__(
#         self, input_width, label_width, shift,
#         batch_size, train_df, val_df, test_df,
#         strides, label_columns = None):

#         self.train_df = train_df
#         self.val_df = val_df
#         self.test_df = test_df
#         self.strides = strides
#         self.label_columns = label_columns
#         if label_columns:
#             self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
#         self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

#         self.input_width = input_width
#         self.label_width = label_width
#         self.shift = shift
#         self.batch_size = batch_size

#         self.total_window_size = input_width + shift
        
#         self.input_slice = slice(0, input_width)
#         self.input_indices = np.arange(self.total_window_size)[self.input_slice]

#         self.label_start = self.total_window_size - self.label_width
#         self.labels_slice = slice(self.label_start, None)
#         self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

#     def __repr__(self):
#         return '\n'.join([
#                 f'Total window size: {self.total_window_size}',
#                 f'Input indices: {self.input_indices}',
#                 f'Label indices: {self.label_indices}',
#                 f'Label column name(s): {self.label_columns}'])

#     def split_window(self, features):
#         inputs = features[:, self.input_slice, :]
#         labels = features[:, self.labels_slice, :]
#         if self.label_columns:
#             labels = tf.stack(
#                 [labels[:, :, self.column_indices[name]] for name in self.label_columns], axis = -1
#             )
#         inputs.set_shape([None, self.input_width, None])
#         labels.set_shape([None, self.label_width, None])
#         return inputs, labels
    
#     def make_dataset(self, data):
#         data = np.array(data, dtype = np.float32)
#         ds = tf.keras.preprocessing.timeseries_dataset_from_array(
#             data = data,
#             targets=None,
#             sequence_length = self.total_window_size,
#             sequence_stride = self.strides,
#             shuffle = True,
#             batch_size = self.batch_size
#         )
#         ds = ds.map(self.split_window)
#         return ds
    
# @property
# def train(self):
#     return self.make_dataset(self.train_df)
# @property
# def val(self):
#     return self.make_dataset(self.val_df)
# @property
# def test(self):
#     return self.make_dataset(self.test_df)
# @property
# def example(self):
#     # find total number of batches in self.test. cardinality() method give the number of batches
#     b_tst = self.test.cardinality().numpy()
#     np.random.seed(3)
#     b_indices = np.random.choice(b_tst, size = 2, replace = False)

#     if not hasattr(self, '_example_cache'):
#         self._example_cache = {}
    
#     if self._n in self._example_cache:
#         return self._example_cache[self._n]
    
#     if self._n == 0:
#         result = next(iter(self.test))
#     elif self._n == 1:
#         result = next(itertools.islice(self.test, b_indices[0], b_indices[0]+1))
#     else:
#         result = next(itertools.islice(self.test, b_indices[1], b_indices[1]+1))

#     self._example_cache[self._n] = result
#     return result


# WindowGenerator.train = train
# WindowGenerator.val = val
# WindowGenerator.test = test
# WindowGenerator.example = example

# def predict_signals(self, model, inputs, outsteps = 375, length_change = False):
#     """
#     model -- trained model
#     inputs -- shape (m, input_width, num_inputs) -- (m, input_width, 12)
#     timesteps -- number of future timesteps
#     length_change -- boolean, whether the input_length at each time step change or static.
#     """
#     predictions = []
#     input_width = inputs.shape[1]
#     dbs_params = inputs[:, :1, :4]
#     curr_input = inputs[:, :, :]
#     for i in range(outsteps):
#         # curr_input = (m, input_width, 12), out = (m, 1, 8)
#         out = model.predict(curr_input)
#         predictions.append(out) # len(prediction) = (i+1) at ith iteration
#         out_reshape = tf.concat([dbs_params, out], axis = -1) # out_reshape = (m, 1, 12)
#         if length_change:
#             curr_input = tf.concat([curr_input, out_reshape], axis = 1) # curr_input = (m,input_width + i + 1, 12 )
#         else:
#             curr_input = tf.concat([curr_input[:, 1:, :], out_reshape], axis = 1) # curr_input = (m,input_width, 12 )

#     return tf.concat(predictions, axis = 1)   

# WindowGenerator.predict_signals = predict_signals

# def plot_channel(self, model = None, name = None, outsteps = 375, plot_col = 'channel1', max_subplots = 3, length_change = True):
#     plt.figure(figsize = (12, 8))
#     # Find the column index
#     plot_col_index = self.column_indices[plot_col]

#     # b_indices = np.random.choice(b_tst, size = 3, replace = False) # randomly choose 3 batches from self.test for plotting
#     for n in range(max_subplots):
#         # retrieve the data
#         self._n = n
#         inputs, labels = self.example

#         # plot inputs (only the first example in each batch)
#         plt.subplot(max_subplots, 1, n+1)
#         plt.ylabel(f'{plot_col} [normed]')
#         print(f"input_indices: {self.input_indices}")
#         plt.plot(self.input_indices, inputs[0, :, plot_col_index],
#              label=f'Inputs_{inputs[0, 0, :4]}', marker='.', markersize=10, zorder=-10)

#         # plot real labels
#         if self.label_columns:
#             label_col_index = self.label_columns_indices.get(plot_col, None)
#         else:
#             label_col_index = plot_col_index

#         if label_col_index is None:
#             continue

#         plt.scatter(self.label_indices, labels[0, :, label_col_index],
#                 label='Labels', c='k', s=32)
#         print(f"label indices: {self.label_indices}")
#         if model is not None:
#             predictions = self.predict_signals(model, inputs[:1, :, :], outsteps = outsteps, length_change = length_change)
#             if predictions.shape[1] > labels.shape[1]:
#                 plt.scatter(self.label_indices, predictions[0, :labels.shape[1], label_col_index],
#                     marker='X', label='Predictions',
#                     c='#ff7f0e', s=32)
#                 forecast_indices = [i for i in range(labels.shape[1]+inputs.shape[1], predictions.shape[1]+inputs.shape[1])]
#                 plt.scatter(forecast_indices, predictions[0, labels.shape[1]:, label_col_index],
#                 marker='X', label='Forecasting', s=32)
#             # print(f"label indices forcast: {self.label_indices}")
#             else:
#                 plt.scatter(self.label_indices, predictions[0, :, label_col_index],
#                     marker='X', label='Predictions',
#                     c='#ff7f0e', s=32)

#         # plt.xlim((-1, 375))
#         # if n == 0:
#         plt.legend()

#     plt.xlabel(plot_col)
#     if name is not None:
#         plt.savefig(os.path.join(save_prediction_images_dir,f"{name}.png"), dpi=300)
# WindowGenerator.plot_channel = plot_channel



# def compute_test_mae(self, model = None, outsteps = 375):
#     """
#     given a model, compute_test_mae() forcasts the values in self.test and evaluate the prediction using mean_absolute_error metric
#     model -- a trained model instance
#     """
#     L = [] #create an empty list to store mae loss for all the batches in self.test
#     data = self.test
#     for inputs, labels in data.take(-1):
#         # data.take(-1) takes all the elements in data or self.test
#         predictions = self.predict_signals(model, inputs, outsteps = outsteps) #using closed loop inference for prediction
#         # print("Labels shape:", labels.shape)
#         # print("Predictions shape:", predictions.shape)
#         l = tf.keras.losses.mean_absolute_error(labels, predictions) # compute the mae for all examples in current batch

#         L .append(tf.math.reduce_mean(l).numpy()) # compute the overall average mae for current batch
#     # return the average mae loss over entire batches in self.test
#     return [np.mean(L)]
# WindowGenerator.compute_test_mae = compute_test_mae







class NormalizedData:
    def __init__(self, norm_type = 'std', input_features=45):
        """
        norm_type -- which normalization method to use: StandardScaler, MinMaxScaler, RobustScaler or PowerTransformer(method = 'yeo-johnson')
        input_features -- default is 45 (all 45 features from the original datasets)

        """
        self.norm_type = norm_type
        self.input_features = input_features

    def flatten_data(self, df, downsample = 1, original=True):
        """
        inputs:
            df -- a list consists of 2D array of shape (m, 1) where m is the number of sessions in df.
                  To access each element in df, we use: session = df[i][0] for i = 0 to m-1. Each session
                  has shape (m_s, 45) where m_s is the length of s session.
        return:
            flatten_data -- a 2D numpy array
            lengths -- a list that store the Txi-length of each element in df. We need this during inversed data step
        """

        flatten_data, lengths = [], []
        for data in df:
            if downsample == 1:
                if not original:
                    lengths.append(data.shape[0])
                    flatten_data.extend(data)
                else:
                    lengths.append(data[0].shape[0]) # store the length of each session in df
                    flatten_data.extend(data[0]) #combine all the sessions into one list
            else:
                temp_list = []
                if not original:
                    for t in range(0, data.shape[0], downsample):
                        temp_list.append(data[t, :])
                else:
                    for t in range(0, data[0].shape[0], downsample):
                        temp_list.append(data[0][t, :])
                temp_list = np.array(temp_list)
                lengths.append(temp_list.shape[0])
                flatten_data.append(temp_list)
                
        flatten_data = (np.array(flatten_data)) #convert to 2D array of shape (sum(lengths), 45)
        return flatten_data[:, :self.input_features], lengths

    def inversed_flatten_data(self, flatten_data, lengths):
        """
        inputs:
            flatten_data -- output from flatten_data(), 2D numpy array
            lengths -- output from flatten_data(), a list
        return:
            inversed_data -- a list consist of inconsistent length 2D numpy array
        """
        inversed_data = []
        start, end = 0, 0
        for l in range(len(lengths)):
            end += lengths[l]
            seq = flatten_data[start:end, :]
            inversed_data.append(seq)
            start += lengths[l]
        return inversed_data
    def mean_norm(self, df):
        """
        Normalization using:
                x = (x- mean)/(max - min)
        inputs:
            df -- a flatten array
        return:
            normalized_data: df after normalization
            cache: mean and range of df 
        
        """
        # compute the mean of the dataset (column wise)
        mu = np.mean(df, axis = 0)
        # compute the range
        df_range = np.max(df, axis = 0) - np.min(df, axis = 0)
        normalized_data = (df - mu) / df_range
        cache = (mu, df_range)
        
        return (normalized_data, cache)

    def normalized_df(self, df, downsample = 1):
        """
        inputs:
            df -- flatten array
        return:
            scaler -- scaler object which is needed for transforming test data
            norm_data -- 2D numpy array (train normalized data)
            lengths -- output from flatten_data(trainData), a list that stores the lengths of all sequences in trainData
        """
        # the normalization function requires the input to be a 2D array. df is a list consist of 2D arrays
        # which can be considered as 3D. So we need to flatten the df before normalization
        # flatten_data, lengths = self.flatten_data(df, downsample = downsample)
        """we consider 4 normalization methods:
        1. std: Standard scaling (x - mean)/sd where sd = standard deviation. After standardization, each feature 
            will have mean 0 and standard deviation 1
        2. mima: MinMax scaling
            X_std = (X - X.min)/(X.max - X.min) followed by X_scaled - X_std*(max-min) + min. After minmax scaling, the range of training set in each feature
            will be between 0 and 1.
        3. robust: Robust scaler
        4. yeo: yeo-johnson scaler
        """
        flatten_data = df
        scalers_mapping = {
            'std': StandardScaler(), 'mima': MinMaxScaler(),
            'robust': RobustScaler(), 'yeo': PowerTransformer(method = 'yeo-johnson')
        }
        if self.norm_type in scalers_mapping:
            scaler = scalers_mapping[self.norm_type]
            norm_data = scaler.fit_transform(flatten_data)
        elif self.norm_type == "mean_normalization":
            norm_data, scaler = self.mean_norm(flatten_data)
            # scaler is a tuple (mean, range)

        return scaler, norm_data
    

class Visualizer:

    def convert_to_dataframe(self, flatten_data, col_names):
        return pd.DataFrame(flatten_data, columns = col_names)

    def plot_dbs(self, data_frame, dbs_names, title_name):
        """

        """
        for name in dbs_names:
            unique_value = data_frame[name].unique()
            data_frame[name].plot(label = f"{name}-{unique_value}")
        plt.legend()
        plt.title(title_name)
    
    def plot_main_feature(self, data_sets, names, channel, set_plot = 0, seq_length = 650*2):
        """
        
        """
        if set_plot == 0:
            df1, df2, df3 = data_sets[0]
            name1, name2, name3 = names[0]
            df1[channel][:seq_length].plot(label = f"{name1}-{df1.iloc[0, 0]}_{df1.iloc[0,1]}")
            df2[channel][:seq_length].plot(label = f"{name2}-{df2.iloc[0, 0]}_{df2.iloc[0,1]}")
            df3[channel][:seq_length].plot(label = f"{name3}-{df3.iloc[0, 0]}_{df3.iloc[0,1]}")
            
        else:
            df1, df2, df3 = data_sets[1]
            name1, name2, name3 = names[1]
            df1[channel][:seq_length].plot(label = f"{name1}-{df1.iloc[0, 0]}_{df1.iloc[0,1]}")
            df2[channel][:seq_length].plot(label = f"{name2}-{df2.iloc[0, 0]}_{df2.iloc[0,1]}")
            df3[channel][:seq_length].plot(label = f"{name3}-{df3.iloc[0, 0]}_{df3.iloc[0,1]}")
        plt.xlabel("steps")
        plt.ylabel(channel)
        plt.legend()
        
    def plot_predict_vs_real(self, predictions, labels, inputs, predict_type):
        """
        plot_predict_vs_real() plots the prediction vs real label on one example
        
        inputs:
          predictions --  2D array of shape (label_width, 8)
          labels -- 2D array of shape (label_width, 8)
          inputs -- 2D array of shape (input_width, 8)
          widths -- tuple (input_width, label_width)
        return:
          a plot that show prediction vs real
        """
        input_width, label_width = inputs.shape[0], labels.shape[0]
        num_channels = predictions.shape[1]
        channels = [f"Channel{str(i)}" for i  in range(num_channels)]
        input_steps = [i for i in range(input_width)]
        output_steps = [i for i in range(input_width, input_width + label_width)]
        plt.figure(figsize = (16, 14))
        for i in range(num_channels):
            plt.subplot(4, 2, i+1)
            plt.plot(input_steps, inputs[:, i], label = 'Inputs')
            plt.plot(output_steps, labels[:, i], label = 'Labels')
            plt.plot(output_steps, predictions[:, i], label = f'Forcasting_{predict_type}')
            plt.legend()
            plt.xlabel('Time Steps')
            plt.ylabel(f'Channel {i+1}')
        plt.tight_layout()
        plt.show()



def split_train_val_test(all_data, file_names, split_ratio):
    """
    all_data: dictionary key:value = file_name:data, data: list of numpy arrays
    file_names: list of file name
    split_ratio: tuple, float, percentage of val, test set
    """
    val_ratio, test_ratio = split_ratio
    train_data = []
    test_data = []
    val_data = []
    for file_name in file_names:
        ## get dataset corresponded to file_name from all_data dictionary
        single_data = all_data[file_name] # a list of numpy arrays of shape (ti, num_desired_col)
        trainsize = int((1 - val_ratio - test_ratio) * len(single_data))
        valsize = int((1 - test_ratio) * len(single_data))
        train_data.extend(single_data[:trainsize])
        val_data.extend(single_data[trainsize:valsize])
        test_data.extend(single_data[valsize:])
    return train_data, val_data, test_data


def flatten_list_array(list_array):
    flatten_data, orig_length = [], []
    for arr in list_array:
        orig_length.append(arr.shape[0])
        flatten_data.extend(arr)
    return orig_length, np.array(flatten_data)

def inversed_flatten_data(flatten_data, orig_length):
    """
    flatten_data -- 2D numpy array - flatten
    orig_length -- list contains length of each city before the data was flatten

    return:
    normed_data: list of numpy arrays
    """
    left, right = 0, 0
    normed_data = []
    for i in range(len(orig_length)):
        # update right
        right += orig_length[i]
        # slice the data from left:right indices
        temp = flatten_data[left:right]
        # add the current time city to normed_data list
        normed_data.append(temp)
        # update left for next iteration
        left += orig_length[i]
    return normed_data
    
            

