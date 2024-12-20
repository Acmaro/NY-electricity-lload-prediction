from functools import partial
import os
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import datetime
import holidays

from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import Dataset, DataLoader

from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from torch.amp import GradScaler, autocast
scaler = GradScaler()

import requests
import zipfile
import matplotlib.pyplot as plt
import urllib
import urllib.request
from datetime import timedelta


def unzip(source_filename,dest_dir):
    with zipfile.ZipFile(source_filename) as zf:
        try:
            zf.extractall(dest_dir)
        except:
            print(source_filename)
            return

def NYISO_load_download(year):
    
    path = f"data/nyiso/{year}/unziped"
    os.makedirs(path, exist_ok=True)
    
    # Download data from NYISO website
    # year = 2023
    dates = pd.date_range(pd.to_datetime(f'{year}-01-01'),pd.to_datetime(f'{year}-12-31'),freq = 'M')
    
    for date in dates:
        url = f'http://mis.nyiso.com/public/csv/pal/{date.year}{date.strftime("%m")}01pal_csv.zip'
        name = url.split('/')[-1]
        print(f"Downloading {name} from {url}")
        urllib.request.urlretrieve(url,f"data/nyiso/{year}/{format(url.split('/')[-1])}")
        # urllib.request.urlretrieve(url,"data/nyiso/{0}".format(url.split('/')[-1]))
        
    # Unzip files
    zips = []
    for file in os.listdir(f'data/nyiso/{year}'):
        if file.endswith('.zip'):
            zips.append(file)
    for z in zips:
        try:
            unzip(f'data/nyiso/{year}/' + z, f'data/nyiso/{year}/unziped')
            print(f'data/nyiso/{year}/' + z, 'extract done')
        except:
            print('data/nyiso/' + z)
            continue
        
    # Merge files
    path = "data/Prepared data"
    os.makedirs(path, exist_ok=True)
    
    path = "data/nyiso/all/"
    os.makedirs(path, exist_ok=True)
    
    csvs = []
    for file in os.listdir(f'data/nyiso/{year}/unziped'):
        if file.endswith('pal.csv'):
            csvs.append(file)
    
    fout = open(f'data/nyiso/all/load_{year}.csv','w')
    for line in open(f'data/nyiso/{year}/unziped/'+csvs[0]):
        fout.write(line)
        # print(line)
    for file in csvs[1:]:
        f = open(f'data/nyiso/{year}/unziped/'+file)
        # print(next(f))
        for line in f:
            fout.write(line)
        f.close()
    fout.close()
    
    df = pd.read_csv(f'data/nyiso/all/load_{year}.csv')

    cols = df.columns
    df.columns = [col.lower().replace(' ','') for col in cols]
    df = df[['timestamp','name','ptid','load']]
    
    region = 'CAPITL'
    subset = df[df.name == region].copy()

    # filename = weather_dict[region][1].lower().replace(' ','')+ '.csv'
    subset.to_csv('data/Prepared data/' + region.lower() + f'_{year}.csv', index = False)

def get_historical_weather_data_of_month(year, month):
    # Get month end date depending on the month
    if month == 2:
        month_end_date = 28
    elif month in [4, 6, 9, 11]:
        month_end_date = 30
    else:
        month_end_date = 31

    month_str = '0' + str(month) if month < 10 else str(month)

    url = f"https://api.weather.com/v1/location/KLGA:9:US/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e&startDate={year}{month_str}01&endDate={year}{month_str}{month_end_date}"

    response = requests.get(url)
    data = response.json()
    return data


def download_weather_data(year):
    interested_fields = ['valid_time_gmt', 'temp', 'wspd', 'pressure', 'precip_hrly']

    with open(f'data/Prepared data/{year}_weather_data.csv', 'w') as f:
        f.write(','.join(interested_fields) + '\n')

        for month in range(1, 13):
            data = get_historical_weather_data_of_month(year, month)
            print(f'Data points for {year}-{month}: {len(data["observations"])}')

            for observation in data['observations']:
                values = []
                for field in interested_fields:
                    if field == 'valid_time_gmt':
                        values.append(datetime.datetime.fromtimestamp(observation[field]).strftime('%Y-%m-%d %H:%M'))
                    else:
                        values.append(str(observation[field]))
                f.write(','.join(values) + '\n')

    print(f'Done')

def is_workday(date:datetime.date):
    """
    Determines the type of day (workday or not) for a given date.

    Args:
        date (datetime.date): The date to check.

    Returns:
        int: 1 if workday else 0.
    """
    us_holidays = holidays.US()

    if date in us_holidays:
        return 0

    if date.weekday() >= 5:  # Saturday is 5, Sunday is 6
        return 0

    return 1

class TimestampTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()

        X['timestamp'] = pd.to_datetime(X['timestamp'])
        X['is_workday'] = X['timestamp'].apply(is_workday)
        return X
    
class TimeSeriesDataset(Dataset):
    def __init__(self, df, seq_len, pred_len):
        """
        Custom Dataset for multivariate time series.

        Args:
            df (pd.Dataframe): Assume the dataframe has been preprocessed and has only numeriacal values.
            seq_length (int): Length of each sequence.
            transform: Composition of transformations.
        """
        super(TimeSeriesDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.features = self.df.values
        self.targets = self.df['load'].values
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.features) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        sequence = self.features[idx:idx + self.seq_len, :]
        sequence = torch.tensor(sequence, dtype=torch.float32)
        
        target = self.targets[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        target = torch.tensor(target, dtype=torch.float32)
        
        return sequence, target

class Preprocessor():
    def __init__(self):
        self.timestamp_transformer = TimestampTransformer()
        self.imputer_load = SimpleImputer(strategy='mean')
        self.imputer_temp = SimpleImputer(strategy='mean')

        self.min_year = None
        
        self.load_mean = None
        self.load_std  = None
        self.temp_mean = None
        self.temp_std  = None
        self.feature_cols = ['load', 'temp', 'is_workday', 'year', 'month', 'day', 'hour', 'minute']

    def fit(self, df):

        self.imputer_load.fit(df[['load']])
        self.imputer_temp.fit(df[['temp']])

        self.min_year = df['year'].min()

        self.load_mean = df['load'].mean()
        self.load_std  = df['load'].std()
        self.temp_mean = df['temp'].mean()
        self.temp_std  = df['temp'].std()

    def transform(self, df):

        df = self.timestamp_transformer.transform(df)

        df['load'] = self.imputer_load.transform(df[['load']])
        df['temp'] = self.imputer_temp.transform(df[['temp']])

        df['year'] = df['year'] - self.min_year

        df['load'] = (df['load'] - self.load_mean) / self.load_std
        df['temp'] = (df['temp'] - self.temp_mean) / self.temp_std

        return df
    
    def inverse_transform(self, df):
        
        df['load'] = df['load'] * self.load_std + self.load_mean

        return df

    def fit_transform(self, df):
        self.fit(df)
        df_out = self.transform(df)
        return df_out

def create_dataset_splits(df_all, preprocessor, start_year, end_year, input_seq_len, output_seq_len, batch_size):
    """
    Specify the year range `start_year` and `end_year` (inclusive) for training and validation data. The end year will be used as the validation data.
    """
    df_all = df_all.sort_values('timestamp').reset_index(drop=True)

    df_all = preprocessor.fit_transform(df_all)

    # save end_year data as .npy
    df_val = df_all[df_all['timestamp'].dt.year == end_year].copy().reset_index(drop=True)
    df_val = df_val.drop('timestamp', axis=1)
    np.save('test_data.npy', df_val.to_numpy())

    if start_year + 1 == end_year:
        df_train = df_all[df_all['timestamp'].dt.year == start_year].copy().reset_index(drop=True)
    elif start_year + 1 < end_year:
        df_train = df_all[(df_all['timestamp'].dt.year >= start_year) & (df_all['timestamp'].dt.year <= end_year-1)].copy().reset_index(drop=True)
    else:
        raise ValueError('Invalid year range.')
    df_val = df_all[df_all['timestamp'].dt.year == end_year].copy().reset_index(drop=True)

    df_train = df_train.drop('timestamp', axis=1)
    df_val = df_val.drop('timestamp', axis=1)

    train_dataset = TimeSeriesDataset(df_train, input_seq_len, output_seq_len)
    val_dataset  = TimeSeriesDataset(df_val, input_seq_len, output_seq_len)

    # For Windows users, set num_workers=0; For Linux users, set num_workers=num_CPU_cores
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, pin_memory_device='cuda')
    val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, pin_memory_device='cuda')

    return train_loader, val_loader

def data_preprocess(year, wd):

    loads = pd.read_csv(f'{wd}/data/Prepared data/capitl_{year}.csv')
    weather = pd.read_csv(f'{wd}/data/Prepared data/{year}_weather_data.csv')
    
    weather = weather.rename(columns={"valid_time_gmt": "timestamp"})
    
    weather['timestamp'] = pd.to_datetime(weather['timestamp'])
    loads['timestamp'] = pd.to_datetime(loads['timestamp'])
    
    def find_nearest(group, match, groupname):
        nbrs = NearestNeighbors(n_neighbors=1).fit(match['timestamp'].values[:, None])
        dist, ind = nbrs.kneighbors(group['timestamp'].values[:, None])
    
        group['nearesttime'] = match['timestamp'].values[ind.ravel()]
        return group
    
    loads = find_nearest(loads, weather,'timestamp')
    
    full = loads.merge(weather, left_on='nearesttime', right_on='timestamp')
    df = full[['timestamp_x', 'load', 'temp']].rename(columns={'timestamp_x': 'timestamp'})
    
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    
    # sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    df.to_csv(f'{wd}/data/Prepared data/{year}_features.csv',encoding='utf-8-sig', index=False)
    return df

# Model Implementation
# --------------------
class SimpleLSTM(nn.Module):
    def __init__(self, num_features, output_size, hidden_size, num_layers, drop_rate=0):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(num_features, hidden_size, num_layers, dropout=drop_rate, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        
        return out

class SimpleGRU(nn.Module):
    def __init__(self, num_features, output_size, hidden_size, num_layers, drop_rate=0):
        super(SimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(num_features, hidden_size, num_layers, dropout=drop_rate, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        
        return out

def model_evaluation(model, criterion, data_loader, device='cpu'):

    batch_losses = []

    model.eval() # switch to evalution mode
    with torch.no_grad():
        for inputs, labels in data_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            batch_losses.append(loss.item())

    model.train() # switch to training mode

    loss_mean = np.mean(batch_losses)

    return loss_mean

def plot_metrics(train_metrics, test_metrics, metric_name):
    plt.figure(figsize=(8, 6))
    epochs = np.arange(len(train_metrics))

    plt.plot(epochs, train_metrics, label=f'Train {metric_name}', color='blue')
    plt.plot(epochs, test_metrics, label=f'Test {metric_name}', color='red')

    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def train_model(config):
    start_year = config['start_year']
    end_year = config['end_year']
    preprocessor = config['preprocessor']
    wd = config['wd'] # working directory
    batch_size = config["batch_size"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    drop_rate = config["drop_rate"]
    lr_rate = config["lr_rate"]
    weight_decay = config["weight_decay"]
    model_type = config["model_type"]

    if model_type == 'LSTM':
        model = SimpleLSTM(num_features=8, output_size=288, hidden_size=hidden_size, num_layers=num_layers, drop_rate=drop_rate)
    elif model_type == 'GRU':
        model = SimpleGRU(num_features=8, output_size=288, hidden_size=hidden_size, num_layers=num_layers, drop_rate=drop_rate)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay, fused=True)

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0
    

    df_list = []
    for year in range(2019, 2022 + 1):
        df = data_preprocess(year, wd)
        df_list.append(df)
    df_raw = pd.concat(df_list, axis=0, ignore_index=True)
    del df_list, df

    df = df_raw[['timestamp', 'load', 'temp', 'year', 'month', 'day', 'hour', 'minute']].copy()
    del df_raw

    # create train and validation data loaders
    train_loader, val_loader = create_dataset_splits(df, preprocessor=preprocessor, start_year=start_year, end_year=end_year, input_seq_len=288, output_seq_len=288, batch_size=batch_size)

    scheduler = None

    for epoch in range(start_epoch, 1):
        model.train()
        for x_batch, y_batch in train_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device=='cuda')):
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        if scheduler is not None:
            scheduler.step()

        # Train/Validation losses
        train_loss = model_evaluation(model, criterion, train_loader, device=device)
        val_loss = model_evaluation(model, criterion, val_loader, device=device)

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"train_loss": train_loss, "val_loss": val_loss},
                checkpoint=checkpoint,
            )

    print("Finished Training")

if __name__ == "__main__":

    # Specify model input and ouput sequence length
    input_seq_len = 288
    output_seq_len = 288

    year_list = [2019, 2020, 2021, 2022, 2023]
    
    print('NYISO load data downloading...')
    for year in year_list:
        NYISO_load_download(year)
    print('Weather data downloading...')
    for year in year_list:
        download_weather_data(year)

    # df_train.to_csv('train_data.csv', index=False)
    # df_test.to_csv('test_data.csv', index=False)

    # init_year = 2015
    # end_year = 2019 # inclusive

    # print('Reading training data...')
    # df_list = []
    # for year in range(init_year, end_year + 1):
    #     df = pd.read_csv(r"C:\Users\zhaor\OneDrive - McMaster University\COMPSCI 4AL3\Final Project\NY-electricity-load-prediction\data\Prepared data\{}_features.csv".format(str(year)))
    #     df_list.append(df)
    # df_train_raw = pd.concat(df_list, axis=0, ignore_index=True)
    # df_train = df_train_raw[['timestamp', 'load', 'temp']].copy()
    # del df_list, df, df_train_raw

    # print('Reading test data...')
    # df_test_raw = pd.read_csv(r"C:\Users\zhaor\OneDrive - McMaster University\COMPSCI 4AL3\Final Project\NY-electricity-load-prediction\data\Prepared data\2020_features.csv")
    # df_test = df_test_raw[['timestamp', 'load', 'temp']].copy()
    # del df_test_raw

    preprocessor = Preprocessor()

    config = {
        "batch_size": tune.choice([32, 64, 128, 256]),
        "hidden_size": tune.choice([16, 32, 64, 128]),
        "num_layers": tune.choice([2, 3, 4, 5]),
        "drop_rate": tune.uniform(0.0, 0.5),
        "lr_rate": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "model_type": tune.choice(["LSTM", "GRU"]),
        "num_gpus": 1,
        "wd": os.getcwd(),
        "preprocessor": preprocessor,
        "start_year": 2019,
        "end_year": 2022
    }

    scheduler = ASHAScheduler(
            metric="val_loss",
            mode="min",
            max_t=30,
            grace_period=10,
            reduction_factor=2,
    )
    result = tune.run(
        partial(train_model),
        resources_per_trial={"cpu": 1, "gpu": 1},
        config=config,
        num_samples=20,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("val_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['val_loss']}")

    # save processed test data as .npy
    # df_list = []
    # for year in range(2019, 2023):
    #     df = pd.read_csv(f'data/Prepared data/{year}_features.csv')
    #     df_list.append(df)
    # df_all = pd.concat(df_list, axis=0, ignore_index=True)
    # preprocessor = Preprocessor()
    # df_all = preprocessor.fit_transform(df_all)
    # df_val = df_all[df_all['timestamp'].dt.year == 2022].copy().reset_index(drop=True)
    # df_val = df_val.drop('timestamp', axis=1)
    # np.save('./data/test_data.npy', df_val.to_numpy())




