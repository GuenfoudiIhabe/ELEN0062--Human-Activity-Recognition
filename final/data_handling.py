import os
import numpy as np
import pandas as pd
from scipy.stats import skew as skewness, kurtosis as excess_kurtosis
from scipy.signal import welch as compute_welch


def compute_metrics(series):
    if np.all(series == series[0]): #Handling flat time series 
        series += np.random.normal(0, 1e-6, size=series.shape) # add a small noise to avoid division by zero

    metrics = {}
    metrics['average'] = np.mean(series)
    metrics['deviation'] = np.std(series)
    metrics['minimum'] = np.min(series)
    metrics['maximum'] = np.max(series)
    metrics['median_val'] = np.median(series) 
    metrics['skew'] = skewness(series, nan_policy='omit') # asymetry coefficient, if skew > 0 the distribution is right skewed elif skew <0 the distribution is left skewed
    metrics['kurt'] = excess_kurtosis(series, nan_policy='omit')# sharpness coefficient of the time series, if kurt > 0 the distribution is more pointed than the normal elif kurt < 0 it is flatter
    metrics['span'] = metrics['maximum'] - metrics['minimum']
    metrics['variance'] = np.var(series)
    metrics['p25'] = np.percentile(series, 25)
    metrics['p75'] = np.percentile(series, 75)

    crossings = np.where(np.diff(np.sign(series)))[0] # count the number of times the time series changes sign
    metrics['zero_cross'] = len(crossings) / len(series)

    metrics['signal_energy'] = np.sum(series ** 2) / len(series) #Mean signal energy defined as the average of the squares of the amplitudes

    frequency, power = compute_welch(series) #returns the frequencies and the associated power by the time series
    if np.sum(power) > 0: #spectral analysis of time series
        metrics['spec_centroid'] = np.sum(frequency * power) / np.sum(power) # it is the sum of the frequency x power / power (center of gravity of the spectrum)
        metrics['spec_bandwidth'] = np.sqrt(
            np.sum(((frequency - metrics['spec_centroid']) ** 2) * power) / np.sum(power) #The spectral width measures the dispersion of frequencies around the spectral centroid. It is an indication of the "width" or "variability" of the spectrum.
        )
    else:
        metrics['spec_centroid'] = 0 #if the total power is zero the spectral metrics are logically zero
        metrics['spec_bandwidth'] = 0
    
    return metrics


def load_and_prepare(directory, dataset='LS'):
    sensor_indices = list(range(2, 33))
    path = os.path.join(directory, dataset)
    feature_frames = []

    for sid in sensor_indices: #Go through each sensor
        if dataset == 'LS':
            print(f"Handling LS sensor {sid} data...") 
        else:
            print(f"Handling TS sensor {sid} data...")


        file_path = os.path.join(path, f'{dataset}_sensor_{sid}.txt')
        data = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python')
        data.replace(-999999.99, np.nan, inplace=True) 

        if dataset == 'LS':
            activity_labels = np.loadtxt(os.path.join(path, 'activity_Id.txt')).astype(int) - 1
            data['activity'] = activity_labels
            data_filled = data.groupby('activity').transform(lambda x: x.fillna(x.mean()))
        else:
            data_filled = data.transform(lambda x: x.fillna(x.mean()))


        feature_list = []
        for _, entry in data_filled.iterrows():  
            time_series = entry.values
            feats = compute_metrics(time_series) #compute the metrics of each series
            feature_list.append(feats) # Stock the metrics in a list

        features = pd.DataFrame(feature_list) 
        features.columns = [f'sensor_{sid}_{col}' for col in features.columns] #Rename the elements by for example sensor_2_average , sensor_3_mean etc...
        feature_frames.append(features)

    features_all = pd.concat(feature_frames, axis=1)

    if dataset == 'LS':
        labels = np.loadtxt(os.path.join(path, 'activity_Id.txt')).astype(int) - 1 # Loading of labels
    else:
        labels = None

    return features_all, labels

def save_data(train_val_X, train_val_y, test_X, directory='./processed_data'):
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    # Save the data
    train_val_X.to_csv(os.path.join(directory, 'train_val_X.csv'), index=False)
    pd.Series(train_val_y).to_csv(os.path.join(directory, 'train_val_y.csv'), index=False, header=False)
    test_X.to_csv(os.path.join(directory, 'test_X.csv'), index=False)
    print(f"Data saved to {directory}")

def load_saved_data(directory='./processed_data'):
    train_val_X = pd.read_csv(os.path.join(directory, 'train_val_X.csv'))
    train_val_y = pd.read_csv(os.path.join(directory, 'train_val_y.csv'), header=None).squeeze("columns").to_numpy()
    test_X = pd.read_csv(os.path.join(directory, 'test_X.csv'))
    print(f"Data loaded from {directory}")
    return train_val_X, train_val_y, test_X