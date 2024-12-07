import datetime
import holidays
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

import os
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import matplotlib.pyplot as plt
import urllib
import urllib.request
import json
import time
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
        X['year'] = X['timestamp'].dt.year
        X['month'] = X['timestamp'].dt.month
        X['day'] = X['timestamp'].dt.day
        X['hour'] = X['timestamp'].dt.hour
        X['minute'] = X['timestamp'].dt.minute
        X = X.drop('timestamp', axis=1)
        return X

class TimeSeriesDataset(Dataset):
    def __init__(self, df, seq_len, pred_len=1, transform=None, target_transform=None):
        """
        Custom Dataset for multivariate time series.

        Args:
            df (pd.Dataframe): Assume the dataframe has been preprocessed and has only numeriacal values.
            seq_length (int): Length of each sequence.
            transform: Composition of transformations.
        """
        super(TimeSeriesDataset, self).__init__()
        self.df = df
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
        self.scaler_load = StandardScaler()
        self.scaler_temp = StandardScaler()

    def fit(self, df):

        self.imputer_load = self.imputer_load.fit(df[['load']])
        self.imputer_temp = self.imputer_temp.fit(df[['temp']])
        self.scaler_load = self.scaler_load.fit(df[['load']])
        self.scaler_temp = self.scaler_temp.fit(df[['temp']])

    def transform(self, df):

        df = self.timestamp_transformer.transform(df)
        df['load'] = self.imputer_load.transform(df[['load']])
        df['temp'] = self.imputer_temp.transform(df[['temp']])
        df['load'] = self.scaler_load.transform(df[['load']])
        df['temp'] = self.scaler_temp.transform(df[['temp']])

        return df

    def fit_transform(self, df, input_seq_len, output_seq_len, batch_size, train_size=0.8):
        
        self.fit(df)
        df = self.transform(df)
        dataset = TimeSeriesDataset(df, input_seq_len, output_seq_len)
        train_size = int(train_size * len(dataset))
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        return train_loader, test_loader

def data_preprocess(year):

    loads = pd.read_csv(f'data/Prepared data/capitl_{year}.csv')
    weather = pd.read_csv(f'data/Prepared data/{year}_weather_data.csv')
    
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
        
    df.to_csv(f'./{year}_features.csv',encoding='utf-8-sig', index=False)
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
    
    # def _initialize_weights(self, num_features):
    #     # According to the paper, https://arxiv.org/pdf/1912.10454
    #     #   we want to preserve the variance through layers.
    #     # As a simplified approach:
    #     # - Set all biases to zero
    #     # - Initialize input-to-hidden weights with a variance ~ 1/N, where N is the number of features
    #     # - Initialize hidden-to-hidden weights orthogonally or with a small variance
        
    #     for name, param in self.lstm.named_parameters():
    #         if 'bias' in name:
    #             nn.init.zeros_(param)
    #         elif 'weight_ih' in name:
    #             # Input to hidden weights: normal with std ~ 1/sqrt(num_features)
    #             nn.init.normal_(param, mean=0.0, std=(1.0 / np.sqrt(num_features)))
    #         elif 'weight_hh' in name:
    #             # Hidden to hidden weights: orthogonal initialization can help stability
    #             nn.init.orthogonal_(param)

    #     # For the fully connected layer
    #     nn.init.zeros_(self.fc.bias)
    #     # Xavier is a reasonable choice for the final layer
    #     nn.init.xavier_normal_(self.fc.weight)

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

def training_loop(n_epochs, optimizer, model, criterion, train_loader, test_loader, verbose=False, scheduler=None, device='cpu', save_model=None, save_as='model.pt'):
    '''
    Set `verbose=True` to see scores for each epoch. If cuda is available, set `device='cuda'`.

    Return
    ------
    - train_losses (list): history of training loss
    - test_losses (list): history of test/validation loss
    '''
    train_losses = []
    test_losses = []

    min_test_loss = float('inf')
    for n in range(n_epochs):
        for x_batch, y_batch in train_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()
        
        if scheduler != None:
            scheduler.step() # update learning rate
        
        train_loss = model_evaluation(model, criterion, train_loader, device=device)
        test_loss = model_evaluation(model, criterion, test_loader, device=device)

        # save model with lowest test/validation loss
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            if save_model == 'best':
                torch.save(model.state_dict(), save_as)

        # save model at last epoch
        if save_model == 'last' and n == n_epochs - 1:
            torch.save(model.state_dict(), save_as)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if ((n + 1) % 10 == 0) or verbose:
            print(f'Epoch {n + 1}/{n_epochs}: Training loss {train_loss:.4f}, Validation Loss {test_loss:.4f}')
            if scheduler != None:
                print(f"Current learning rate is {scheduler.get_last_lr()[0]}")
            print('----------------------------------------------------------')

    return train_losses, test_losses

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

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Parameters for data preprocessing
    input_seq_len = 288
    output_seq_len = 288
    batch_size = 128
    # Parameters for model initialization and training
    num_features = 8
    hidden_size = 32
    num_layers = 2
    num_epochs = 30
    drop_rate = 0.3
    lr_rate = 1e-4
    weight_decay = 1e-4

    year_list = [2015, 2016, 2017, 2018, 2019, 2020]
    
    print('NYISO load data downloading...')
    for year in year_list:
        NYISO_load_download(year)
    print('Weather data downloading...')
    for year in year_list:
        download_weather_data(year)
    print('Combining load and weather data...')
    
    df_list = []
    for year in year_list:
        df = data_preprocess(year)
        df_list.append(df)

    df_train = pd.concat(df_list[:-1], axis=0, ignore_index=True)
    df_test = df_list[-1]
    del df_list, df

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

    print('Preprocessing data...')
    processing = Preprocessor()
    train_loader, val_loader = processing.fit_transform(
        df=df_train,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        batch_size=batch_size
    )

    print('Saving test data...')
    df_test = processing.transform(df_test)
    test_np = df_test.to_numpy(dtype=np.float32)
    np.save('test_data.npy', test_np) # save test data for later use
    del test_np, df_test

    print('Initializing models and training...')
    lstm_model = SimpleLSTM(num_features, output_seq_len, hidden_size, num_layers, drop_rate=drop_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    train_losses, val_losses = training_loop(
        num_epochs, 
        optimizer, 
        lstm_model, 
        criterion, 
        train_loader, 
        val_loader, 
        verbose=True, 
        scheduler=scheduler,
        device=device, 
        save_model='last',
        save_as='LSTM.pt'
    )
    print('LSTM model training completed, model saved as LSTM.pt.')
    plot_metrics(train_losses, val_losses, 'Loss')

    gru_model = SimpleGRU(num_features, output_seq_len, hidden_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(gru_model.parameters(), lr=lr_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    train_losses, val_losses = training_loop(
        num_epochs, 
        optimizer, 
        gru_model, 
        criterion, 
        train_loader, 
        val_loader, 
        verbose=True, 
        scheduler=scheduler,
        device='cuda', 
        save_model='last',
        save_as='GRU.pt'
    )
    print('GRU model training completed, model saved as GRU.pt.')
    plot_metrics(train_losses, val_losses, 'Loss')




