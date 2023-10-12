# This cell includes the major classes used in our classification analyses
import matplotlib.pyplot as plt
import pywt
from mpl_toolkits.mplot3d import Axes3D # for 3D plotting, woohoo!
import numpy as np
import scipy as sp
from scipy import signal
from scipy.stats import skew, kurtosis
import random
import os
import math
import itertools
from IPython.display import display_html

# We wrote this gesturerec package for the class
# It provides some useful data structures for the accelerometer signal
# and running experiments so you can focus on writing classification code,
# evaluating your solutions, and iterating
import inductancerec.utility as dfutils
import inductancerec.data as dfdata
import inductancerec.vis as dfvis

from inductancerec.data import SensorData
from inductancerec.data import DeformationSet

# Scikit-learn stuff
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict, StratifiedKFold

import pandas as pd
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
import time

def display_tables_side_by_side(df1, df2, n=None, df1_caption="Caption table 1", df2_caption="Caption table 2"):
    '''Displays the two tables side-by-side'''

    if n is not None:
        df1 = df1.head(n)
        df2 = df2.head(n)

    # Solution from https://stackoverflow.com/a/50899244
    df1_styler = df1.style.set_table_attributes("style='display:inline; margin:10px'").set_caption(df1_caption)
    df2_styler = df2.style.set_table_attributes("style='display:inline'").set_caption(df2_caption)

    display_html(df1_styler._repr_html_() + df2_styler._repr_html_(), raw=True)


def print_folds(cross_validator, X, y_true, trial_indices):
    '''Prints out the k-fold splits'''
    fold_cnt = 0
    for train_index, test_index in cross_validator.split(X, y_true):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_true.iloc[train_index], y_true.iloc[test_index]
        print("TEST FOLD {}".format(fold_cnt))
        for i in test_index:
            print("\t{} {}".format(y_true[i], trial_indices[i]))
        fold_cnt += 1


def display_folds(cross_validator, X, y_true, trial_indices):
    map_fold_to_class_labels = dict()
    fold_cnt = 0
    for train_index, test_index in cross_validator.split(X, y_true):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_true.iloc[train_index], y_true.iloc[test_index]

        class_labels = []
        for i in test_index:
            class_labels.append(f"{y_true[i]} {trial_indices[i]}")

        map_fold_to_class_labels[f"Fold {fold_cnt}"] = class_labels
        fold_cnt += 1

    df = pd.DataFrame(map_fold_to_class_labels)
    display_folds(df)


def compute_fft(s, sampling_rate, n=None, scale_amplitudes=True):
    '''Computes an FFT on signal s using numpy.fft.fft.

       Parameters:
        s (np.array): the signal
        sampling_rate (num): sampling rate
        n (integer): If n is smaller than the length of the input, the input is cropped. If n is
            larger, the input is padded with zeros. If n is not given, the length of the input signal
            is used (i.e., len(s))
        scale_amplitudes (boolean): If true, the spectrum amplitudes are scaled by 2/len(s)
    '''
    if n == None:
        n = len(s)

    fft_result = np.fft.fft(s, n)
    num_freq_bins = len(fft_result)
    fft_freqs = np.fft.fftfreq(num_freq_bins, d=1 / sampling_rate)
    half_freq_bins = num_freq_bins // 2

    fft_freqs = fft_freqs[:half_freq_bins]
    fft_result = fft_result[:half_freq_bins]
    fft_amplitudes = np.abs(fft_result)

    if scale_amplitudes is True:
        fft_amplitudes = 2 * fft_amplitudes / (len(s))

    return (fft_freqs, fft_amplitudes)


def get_top_n_frequency_peaks(n, freqs, amplitudes, min_amplitude_threshold=None):
    ''' Finds the top N frequencies and returns a sorted list of tuples (freq, amplitudes) '''

    # Use SciPy signal.find_peaks to find the frequency peaks
    fft_peaks_indices, fft_peaks_props = sp.signal.find_peaks(amplitudes, height=min_amplitude_threshold)

    freqs_at_peaks = freqs[fft_peaks_indices]
    amplitudes_at_peaks = amplitudes[fft_peaks_indices]

    if n < len(amplitudes_at_peaks):
        ind = np.argpartition(amplitudes_at_peaks, -n)[-n:]  # from https://stackoverflow.com/a/23734295
        ind_sorted_by_coef = ind[np.argsort(-amplitudes_at_peaks[ind])]  # reverse sort indices
    else:
        ind_sorted_by_coef = np.argsort(-amplitudes_at_peaks)

    return_list = list(zip(freqs_at_peaks[ind_sorted_by_coef], amplitudes_at_peaks[ind_sorted_by_coef]))
    return return_list


map_marker_to_desc = {
    ".": "point",
    ",": "pixel",
    "o": "circle",
    "v": "triangle_down",
    "^": "triangle_up",
    "<": "triangle_left",
    ">": "triangle_right",
    "1": "tri_down",
    "2": "tri_up",
    "3": "tri_left",
    "4": "tri_right",
    "8": "octagon",
    "s": "square",
    "p": "pentagon",
    "*": "star",
    "h": "hexagon1",
    "H": "hexagon2",
    "+": "plus",
    "D": "diamond",
    "d": "thin_diamond",
    "|": "vline",
    "_": "hline"
}

plot_markers = ['o', 'v', '^', '<', '>', 's', 'p', 'P', '*', 'h', 'X', 'D', 'd', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                10, '1', '2', '3', '4', ',']


def plot_feature_1d(deformation_set, extract_feature_func, title=None, use_random_y_jitter=True,
                    xlim=None):
    '''
    Plots the extracted feature on a 1-dimensional plot. We use a random y-jitter
    to make the values more noticeable

    Parameters:

    deformation_set: the DeformationSet class
    extract_feature_func: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature
    title: the graph title
    use_random_y_jitter: provides a random y jitter to make it easier to see values
    xlim: set the x range of the graph
    '''
    markers = list(map_marker_to_desc.keys())
    random.Random(3).shuffle(markers)
    marker = itertools.cycle(markers)
    plt.figure(figsize=(12, 3))
    for deformation_name in deformation_set.get_deformation_names_sorted():
        trials = deformation_set.map_deformations_to_trials[deformation_name]
        x = list(extract_feature_func(trial.inductance) for trial in trials)
        y = None

        if use_random_y_jitter:
            y = np.random.rand(len(x))
        else:
            y = np.zeros(len(x))

        marker_sizes = [200] * len(x)  # make the marker sizes larger
        plt.scatter(x, y, alpha=0.65, marker=next(marker),
                    s=marker_sizes, label=deformation_name)

    plt.ylim((0, 3))

    if xlim is not None:
        plt.xlim(xlim)

    if use_random_y_jitter:
        plt.ylabel("Ignore the y-axis")

    plt.legend(bbox_to_anchor=(1, 1))

    if title is None:
        title = f"1D plot of {extract_feature_func.__name__}"

    plt.title(title)
    plt.show()


def plot_feature_2d(deformation_set, extract_feature_func1, extract_feature_func2,
                    xlabel="Feature 1", ylabel="Feature 2",
                    title=None, xlim=None):
    '''
    Plots the two extracted features on a 2-dimensional plot.

    Parameters:

    deformation_set: the DeformationSet class
    extract_feature_func1: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature
    extract_feature_func2: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature
    title: the graph title
    xlim: set the x range of the graph
    '''
    markers = list(map_marker_to_desc.keys())
    random.Random(3).shuffle(markers)
    marker = itertools.cycle(markers)
    plt.figure(figsize=(12, 5))
    for deformation_name in deformation_set.get_deformation_names_sorted():
        trials = deformation_set.map_deformations_to_trials[deformation_name]
        x = list(extract_feature_func1(trial.inductance) for trial in trials)
        y = list(extract_feature_func2(trial.inductance) for trial in trials)

        marker_sizes = [200] * len(x)  # make the marker sizes larger
        plt.scatter(x, y, alpha=0.65, marker=next(marker),
                    s=marker_sizes, label=deformation_name)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xlim is not None:
        plt.xlim(xlim)

    plt.legend(bbox_to_anchor=(1, 1))

    plt.title(title)
    plt.show()


def plot_feature_3d(deformation_set, extract_feature_func1, extract_feature_func2,
                    extract_feature_func3, xlabel="Feature 1", ylabel="Feature 2",
                    zlabel="Feature 2", title=None, figsize=(12, 9)):
    '''
    Plots the three extracted features on a 3-dimensional plot.

    Parameters:

    deformation_set: the DeformationSet class
    extract_feature_func1: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature
    extract_feature_func2: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature
    extract_feature_func3: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature
    title: the graph title
    xlim: set the x range of the graph
    '''
    markers = list(map_marker_to_desc.keys())
    random.Random(3).shuffle(markers)
    marker = itertools.cycle(markers)
    fig = plt.figure(figsize=figsize)
    #     ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')

    for deformation_name in deformation_set.get_deformation_names_sorted():
        trials = deformation_set.map_deformations_to_trials[deformation_name]
        x = list(extract_feature_func1(trial.inductance) for trial in trials)
        y = list(extract_feature_func2(trial.inductance) for trial in trials)
        z = list(extract_feature_func3(trial.inductance) for trial in trials)

        marker_sizes = [200] * len(x)  # make the marker sizes larger
        ax.scatter(x, y, z, alpha=0.65, marker=next(marker),
                   s=marker_sizes, label=deformation_name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    ax.legend()

    ax.set_title(title)
    plt.show()
    return fig, ax


def plot_bar_graph(d, title=None, ylabel=None, xlabel=None):
    '''
    Plots a bar graph of of the values in d (with the keys as names)
    '''

    sorted_tuple_list = sorted(d.items(), key=lambda x: x[1])
    n_groups = len(d)

    sorted_keys = []
    sorted_values = []
    for k, v in sorted_tuple_list:
        sorted_keys.append(k)
        sorted_values.append(v)

    # create plot
    fig_height = max(n_groups * 0.5, 5)
    plt.figure(figsize=(12, fig_height))
    indices = np.arange(len(sorted_keys))

    plt.grid(zorder=0)
    bars = plt.barh(indices, sorted_values, alpha=0.8, color='b', zorder=3)

    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.xlim(0, sorted_values[-1] * 1.1)
    plt.title(title)
    plt.yticks(indices, sorted_keys)

    for i, v in enumerate(sorted_values):
        plt.text(v + 0.01, i, "{:0.2f}".format(v), color='black', fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_signals(deformation_set, signal_var_names=['ind_data']):
    '''Plots the deformation set as a grid given the signal_var_names'''
    num_rows = len(deformation_set.map_deformations_to_trials)
    num_cols = len(signal_var_names)
    row_height = 3.5
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, row_height * num_rows))
    fig.subplots_adjust(hspace=0.5)

    index = 0
    for row, deformation_name in enumerate(deformation_set.get_deformation_names_sorted()):
        deformation_trials = deformation_set.get_trials_for_deformation(deformation_name)

        for trial in deformation_trials:
            for col, signal_var_name in enumerate(signal_var_names):
                s = getattr(trial.inductance, signal_var_name)
                axes[row][col].plot(s, alpha=0.7, label=f"Trial {trial.trial_num}")

                axes[row][col].set_title(f"{deformation_name}: {signal_var_name}")
                axes[row][col].legend()

    fig.tight_layout(pad=2)
    plt.show()


def plot_signals_aligned(deformation_set, signal_var_names=['ind_data'], title_fontsize=8):
    '''Aligns each signal using cross correlation and then plots them'''
    num_rows = len(deformation_set.map_deformations_to_trials)
    num_cols = len(signal_var_names)
    row_height = 3.5
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, row_height * num_rows))
    fig.subplots_adjust(hspace=0.5)

    index = 0
    for row, deformation_name in enumerate(deformation_set.get_deformation_names_sorted()):
        deformation_trials = deformation_set.get_trials_for_deformation(deformation_name)

        for col, signal_var_name in enumerate(signal_var_names):

            # Find the maximum length signal. We need this to pad all
            # signals to the same length (a requirement of cross correlation)
            signal_lengths = []
            for trial in deformation_trials:
                s = getattr(trial.inductance, signal_var_name)
                signal_lengths.append(len(s))
            max_trial_length = np.max(signal_lengths)

            # Actually pad the signals
            signals = []
            for trial in deformation_trials:
                s = getattr(trial.inductance, signal_var_name)

                padding_length = max_trial_length - len(s)
                if padding_length > 0:
                    pad_amount_left = (math.floor(padding_length / 2.0))
                    pad_amount_right = (math.ceil(padding_length / 2.0))
                    padded_s = np.pad(s, (pad_amount_left, pad_amount_right), mode='mean')
                    signals.append(padded_s)
                else:
                    signals.append(s)

            # Grab a signal to align everything to. We could more carefully choose
            # this signal to be the closest signal to the average aggregate... but this
            # should do for now
            golden_signal = signals[0]  # the signal to align everything to

            # Align all the signals and store them in aligned_signals
            aligned_signals = [golden_signal]
            for i in range(1, len(signals)):
                a = golden_signal
                b = signals[i]
                correlate_result = np.correlate(a, b, 'full')
                best_correlation_index = np.argmax(correlate_result)
                shift_amount = (-len(a) + 1) + best_correlation_index
                b_shifted_mean_fill = shift_array(b, shift_amount, np.mean(b))
                aligned_signals.append(b_shifted_mean_fill)

            # Plot the aligned signals
            for signal_index, trial in enumerate(deformation_trials):
                s = aligned_signals[signal_index]
                axes[row][col].plot(s, alpha=0.7, label=f"Trial {trial.trial_num}")

                axes[row][col].set_title(f"Aligned {deformation_name}: {signal_var_name}", fontsize=title_fontsize)
                axes[row][col].legend()

    fig.tight_layout(pad=2)


def shift_array(arr, shift_amount, fill_value=np.nan):
    '''Shifts the array either left or right by the shift_amount (which can be negative or positive)

       From: https://stackoverflow.com/a/42642326
    '''
    result = np.empty_like(arr)
    if shift_amount > 0:
        result[:shift_amount] = fill_value
        result[shift_amount:] = arr[:-shift_amount]
    elif shift_amount < 0:
        result[shift_amount:] = fill_value
        result[:shift_amount] = arr[-shift_amount:]
    else:
        result[:] = arr
    return result


def plot_fft_signals(deformation_set, signal_var_names=['ind_data']):
    '''Plots the FFT of deformation set as a grid given the signal_var_names'''
    num_rows = len(deformation_set.map_deformations_to_trials)
    num_cols = len(signal_var_names)
    row_height = 3.5
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, row_height * num_rows))
    fig.subplots_adjust(hspace=0.5)

    index = 0
    for row, deformation_name in enumerate(deformation_set.get_deformation_names_sorted()):
        deformation_trials = deformation_set.get_trials_for_deformation(deformation_name)

        for trial in deformation_trials:
            for col, signal_var_name in enumerate(signal_var_names):
                s = getattr(trial.inductance, signal_var_name)
                sampling_rate = getattr(trial.inductance, 'sampling_rate')
                freqs, amplitudes = compute_fft(s, sampling_rate)

                axes[row][col].plot(freqs, amplitudes, alpha=0.7, label=f"Trial {trial.trial_num}")

                axes[row][col].set_title(f"{deformation_name}: FFT of {signal_var_name}")
                axes[row][col].legend()

    fig.tight_layout(pad=2)
    plt.show()


def calculate_sd(inductance):
    s = getattr(inductance, 'ind_data_p')
    return np.sqrt(np.var(s))


def calculate_rms(inductance):
    s = getattr(inductance, 'ind_data_p')
    squared_values = np.square(s)
    mean_squared = np.mean(squared_values)
    rms = np.sqrt(mean_squared)
    return rms


def calculate_skewness(inductance):
    s = getattr(inductance, 'ind_data_p')
    return skew(s)


def calculate_kurtosis(inductance):
    s = getattr(inductance, 'ind_data_p')
    return kurtosis(s)


def calculate_mean_energy(inductance):
    s = getattr(inductance, 'ind_data_p')
    sampling_rate = getattr(inductance, 'sampling_rate')
    freqs, amplitudes = compute_fft(s, sampling_rate)
    return np.mean(amplitudes ** 2)


def dominant_freq(inductance):
    s = getattr(inductance, 'ind_data_p')
    sampling_rate = getattr(inductance, 'sampling_rate')
    freqs, amplitudes = compute_fft(s, sampling_rate)
    max_amplitude_index = np.argmax(amplitudes)
    dominant_frequency = freqs[max_amplitude_index]
    return dominant_frequency


def bandwidth(inductance):
    s = getattr(inductance, 'ind_data_p')
    sampling_rate = getattr(inductance, 'sampling_rate')
    freqs, amplitudes = compute_fft(s, sampling_rate)
    max_amplitude = np.max(amplitudes)
    max_amplitude_index = np.argmax(amplitudes)
    # Define the threshold percentage
    threshold_percent = 0.2
    # Calculate the threshold
    threshold = threshold_percent * max_amplitude

    lower_index = max_amplitude_index
    upper_index = max_amplitude_index
    while amplitudes[lower_index] > threshold and lower_index > 0:
        lower_index -= 1
    while amplitudes[upper_index] > threshold and upper_index < len(amplitudes) - 1:
        upper_index += 1

    frequency_bandwidth = freqs[upper_index] - freqs[lower_index]
    return frequency_bandwidth


def compute_spectral_kurtosis(inductance):
    s = getattr(inductance, 'ind_data_p')
    sampling_rate = getattr(inductance, 'sampling_rate')
    freqs, amplitudes = compute_fft(s, sampling_rate)
    spectrum = amplitudes ** 2
    # Normalization
    normalized_spectrum = spectrum / np.sum(spectrum)

    fourth_moment = kurtosis(normalized_spectrum, fisher=False)
    variance = np.var(normalized_spectrum)

    spectral_kurtosis = (fourth_moment / (variance ** 2)) - 3
    return spectral_kurtosis


def spectral_mean(inductance):
    s = getattr(inductance, 'ind_data_p')
    sampling_rate = getattr(inductance, 'sampling_rate')
    freqs, amplitudes = compute_fft(s, sampling_rate)
    return np.mean(amplitudes)


def spectral_sd(inductance):
    s = getattr(inductance, 'ind_data_p')
    sampling_rate = getattr(inductance, 'sampling_rate')
    freqs, amplitudes = compute_fft(s, sampling_rate)
    return np.std(amplitudes)


def min_index(inductance):
    s = getattr(inductance, 'ind_data_p')
    sampling_rate = getattr(inductance, 'sampling_rate')
    freqs, amplitudes = compute_fft(s, sampling_rate)
    return np.argmin(amplitudes)


def max_index(inductance):
    s = getattr(inductance, 'ind_data_p')
    sampling_rate = getattr(inductance, 'sampling_rate')
    freqs, amplitudes = compute_fft(s, sampling_rate)
    return np.argmax(amplitudes)


def center_of_mass(inductance):
    s = getattr(inductance, 'ind_data_p')
    sampling_rate = getattr(inductance, 'sampling_rate')
    freqs, amplitudes = compute_fft(s, sampling_rate)
    return np.sum(freqs * amplitudes) / np.sum(amplitudes)


def compute_spectral_centroid(inductance):
    s = getattr(inductance, 'ind_data_p')
    sampling_rate = getattr(inductance, 'sampling_rate')
    freqs, amplitudes = compute_fft(s, sampling_rate)

    normalized_amplitudes = amplitudes / np.sum(amplitudes)

    spectral_centroid = np.sum(freqs * normalized_amplitudes)

    return spectral_centroid


def preprocess_signal(s, wavelet='db1', level=1, threshold_type='soft'):
    '''Preprocesses the signal'''

    # Basic algorithm: a mean filter of window size 3.
    # Advanced algorithm: explore detrending and diff filtering algs (with different window sizes)

    mean_filter_window_size = 3
    processed_signal = np.convolve(s,
                                   np.ones((mean_filter_window_size,)) / mean_filter_window_size,
                                   mode='valid')
    return processed_signal



def preprocess_trial(trial):
    '''Processess the given trial'''
    trial.inductance.ind_data_p = preprocess_signal(trial.inductance.ind_data)


def generate_kfolds_scikit(num_folds, deformation_set, seed=None):
    '''
    Here's an example of generating kfolds using scikit but returning our data structure

    Parameters:
    num_folds: the number of folds
    deformation_set: the deformation set for splitting into k-folds
    seed: an integer seed value (pass in the same seed value to get the same split across multiple executions)

    Returns:
    Returns a list of folds where each list item is a dict() with key=deformation name and value=selected trial
    for that fold. To generate the same fold structure, pass in the same seed value (this is useful for
    setting up experiments). Note that even with the same seed value, this method and generate_kfolds will
    generate different results.
    '''

    trials = []
    trial_nums = []
    deformation_names = []
    for deformation_name, deformation_trials in deformation_set.map_deformations_to_trials.items():
        for trial in deformation_trials:
            trials.append(trial)
            trial_nums.append(trial.trial_num)
            deformation_names.append(deformation_name)

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    # Iterate through the splits and setup our data structure
    fold_cnt = 0
    list_folds = list()
    for train_idx, test_idx in skf.split(trials, deformation_names):
        cur_fold_map_deformation_to_trial = dict()
        for i in test_idx:
            cur_fold_map_deformation_to_trial[deformation_names[i]] = trials[i]
        list_folds.append(cur_fold_map_deformation_to_trial)
        fold_cnt += 1
    return list_folds


def print_folds(list_folds):
    '''
    Prints out the folds (useful for debugging)
    '''
    # print out folds (for debugging)
    fold_index = 0
    if fold_index == 0:
        for fold in list_folds:
            print("Fold: ", fold_index)
            for deformation_name, trial in fold.items():
                print("\t{} Trial: {}".format(deformation_name, trial.trial_num))
            fold_index = fold_index + 1


def check_folds(folds):
    '''
    Checks to see that the folds are appropriately setup (useful for debugging)
    Throw an exception if there appears to be a problem
    '''
    for test_fold_idx in range(0, len(folds)):
        # check to make sure test data is not in training data
        for test_deformation, test_trial in folds[test_fold_idx].items():
            # search for this test_deformation and trial_num in all other folds
            # it shouldn't be there!
            for train_fold_idx in range(0, len(folds)):
                if test_fold_idx != train_fold_idx:
                    for train_deformation, train_trial in folds[train_fold_idx].items():
                        if test_deformation == train_deformation and test_trial.trial_num == train_trial.trial_num:
                            raise Exception("Uh oh, deformation '{}' trial '{}' was found in both test fold '{}' and\
                                             training fold '{}.' Training folds should not include test data".format(
                                test_deformation, test_trial.trial_num, test_fold_idx, train_fold_idx))


def extract_feature_from_trial1(trial, extract_feature_func1):
    '''
    This function serves the ability to extract features from a certain fold using three feature extraction functions

    fold : the fold that needs to extract features
    extract_feature_func1: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature

    '''
    result = list()
    result.append(extract_feature_func1(trial.inductance))
    return result


def extract_feature_from_trial2(trial, extract_feature_func1, extract_feature_func2):
    '''
    This function serves the ability to extract features from a certain fold using three feature extraction functions

    fold : the fold that needs to extract features
    extract_feature_func1: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature
    extract_feature_func2: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature
    '''
    result = list()
    result.append(extract_feature_func1(trial.inductance))
    result.append(extract_feature_func2(trial.inductance))
    return result


def extract_feature_from_trial(trial, extract_feature_func1, extract_feature_func2, extract_feature_func3):
    '''
    This function serves the ability to extract features from a certain fold using three feature extraction functions

    fold : the fold that needs to extract features
    extract_feature_func1: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature
    extract_feature_func2: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature
    extract_feature_func3: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature
    '''
    result = list()
    result.append(extract_feature_func1(trial.inductance))
    result.append(extract_feature_func2(trial.inductance))
    result.append(extract_feature_func3(trial.inductance))
    return result


def extract_feature_from_trial4(trial, extract_feature_func1, extract_feature_func2, extract_feature_func3,
                                extract_feature_func4):
    '''
    This function serves the ability to extract features from a certain fold using three feature extraction functions

    fold : the fold that needs to extract features
    extract_feature_func1: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature
    extract_feature_func2: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature
    extract_feature_func3: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature
    extract_feature_func4: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature
    '''
    result = list()
    result.append(extract_feature_func1(trial.inductance))
    result.append(extract_feature_func2(trial.inductance))
    result.append(extract_feature_func3(trial.inductance))
    result.append(extract_feature_func4(trial.inductance))
    return result


def extract_feature_from_trial5(trial, extract_feature_func1, extract_feature_func2, extract_feature_func3,
                                extract_feature_func4, extract_feature_func5):
    '''
    This function serves the ability to extract features from a certain fold using three feature extraction functions

    fold : the fold that needs to extract features
    extract_feature_func1: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature
    extract_feature_func2: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature
    extract_feature_func3: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature
    extract_feature_func4: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature
    extract_feature_func5: a "pointer" to a function that accepts a trial.inductance object and returns an extracted feature
    '''
    result = list()
    result.append(extract_feature_func1(trial.inductance))
    result.append(extract_feature_func2(trial.inductance))
    result.append(extract_feature_func3(trial.inductance))
    result.append(extract_feature_func4(trial.inductance))
    result.append(extract_feature_func5(trial.inductance))
    return result


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Calculate Confusion Matrix
    confusion_mat = confusion_matrix(y_test, y_pred)

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate Macro precision rate
    precision_macro = precision_score(y_test, y_pred, average='macro')

    # Calculate Macro Recall
    recall_macro = recall_score(y_test, y_pred, average='macro')

    # Calculate Macro F1
    f1_macro = f1_score(y_test, y_pred, average='macro')

    metrics = {
        "Accuracy": accuracy,
        "Precision (Macro)": precision_macro,
        "Recall (Macro)": recall_macro,
        "F1 Score (Macro)": f1_macro,
    }

    return metrics, confusion_mat


def plot_confusion_matrix(cm, classes, title, normalize=False, cmap=plt.cm.Blues, custom_text=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")

    # print(cm)
    plt.figure(figsize=(12, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  # imshow displays data on a 2D raster
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    if custom_text:
        plt.text(0.3 * plt.gca().get_xlim()[0], 0.05 * plt.gca().get_ylim()[1], custom_text, fontsize=10, ha='right',
                 va='bottom', color='red')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_evaluation_metric(metric_names, metric_values, title):
    # Calculate the ratio relative to 1
    metric_ratios = [value if isinstance(value, list) else value / 1 for value in metric_values]

    # Create Bar Chart
    fig, ax = plt.subplots()
    bars = ax.bar(metric_names, metric_ratios)

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Values")

    # Set the y-axis scale range
    ax.set_ylim(0, 1.2)
    ax.axhline(y=1, color='red', linestyle='--')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, round(height, 2), ha='center', va='bottom')

    plt.show()