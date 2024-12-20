import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from training import SimpleLSTM, SimpleGRU
from training import Preprocessor

def evaluate_metrics(real_values, predictions): 
    ''' 
    Calculate evaluation metrics: MAE, MAPE, and R-squared. 

    Parameters 
    ---------- 

    real_values (array-like): The actual values. 
    predictions (array-like): The predicted values. 

    Returns 
    ------- 
    mae, mape, r_squared
    ''' 

    real_values = np.array(real_values) 
    predictions = np.array(predictions) 

    # Mean Absolute Error (MAE) 
    mae = np.mean(np.abs(real_values - predictions)) 

    # Mean Absolute Percentage Error (MAPE) 
    epsilon = 1e-7
    mape = np.mean(np.abs((real_values - predictions) / (real_values + epsilon))) * 100

    # R-squared 
    ss_res = np.sum((real_values - predictions) ** 2) 
    ss_tot = np.sum((real_values - np.mean(real_values)) ** 2) 
    r_squared = 1 - (ss_res / ss_tot)

    return mae, mape, r_squared

def make_predictions(model, test_data, device='cuda'):
    """
    Evaluate the model on test_data for the year 2020 and produce 1D arrays of predictions and actuals.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model that takes an input of shape [1, 288, 7] and outputs [1, 288].
    test_data : torch.Tensor
        A tensor of shape [N, 7] containing continuous test data.
        The columns are [load, temp, is_workday, year, month, day, hour].
        Assume test_data is normalized/processed the same way as the training data.
    device : str, optional
        Device to run computations on ('cuda' or 'cpu'), by default 'cuda'.

    Returns
    -------
    predictions : np.ndarray
        1D array containing the model's predicted load values for all predicted timesteps.
    actuals : np.ndarray
        1D array containing the actual load values for the corresponding timesteps.
    """

    model = model.to(device)
    model.eval()

    total_steps = test_data.shape[0]
    # Each prediction uses 288 steps as input and predicts the next 288 steps.
    # Need at least 288 steps beyond the input window for a full prediction.
    max_start = total_steps - 288 * 2
    if max_start < 0:
        raise ValueError("Not enough test data to form a single input-output pair.")

    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        # Slide over test_data in increments of 288 steps (e.g., one day at a time, if 288 steps = one day)
        for start_idx in range(0, max_start + 1, 288):
            # Extract input sequence
            input_seq = test_data[start_idx : start_idx+288]  # [288, 7]
            input_seq = input_seq.unsqueeze(0).to(device)     # [1, 288, 7]

            # Run the model
            pred = model(input_seq)         # [1, 288]
            pred = pred.squeeze(0).cpu().numpy()  # [288]

            # Actual load for the next 288 steps
            target_seq = test_data[start_idx+288 : start_idx+576, 0].cpu().numpy()  # [288]

            # Collect results
            all_predictions.append(pred)
            all_actuals.append(target_seq)

    # Convert to arrays of shape [M, 288]
    all_predictions = np.array(all_predictions)  # [M, 288]
    all_actuals = np.array(all_actuals)          # [M, 288]

    # Flatten to 1D arrays
    predictions = all_predictions.flatten()  # 1D array
    actuals = all_actuals.flatten()          # 1D array

    return predictions, actuals

if __name__ == "__main__":

    # Parameters for data preprocessing
    input_seq_len = 288
    output_seq_len = 288

    # Load test data
    df_list = []
    for year in range(2022, 2023):
        df = pd.read_csv(f'data/Prepared data/{year}_features.csv')
        df_list.append(df)
    df_all = pd.concat(df_list, axis=0, ignore_index=True)
    preprocessor = Preprocessor()
    df_all = preprocessor.fit_transform(df_all)
    df_val = df_all[df_all['timestamp'].dt.year == 2022].copy().reset_index(drop=True)
    df_val = df_val.drop('timestamp', axis=1)
    test_tensor = torch.tensor(df_val.values, dtype=torch.float32)
    # test_np = np.load('data/test_data.npy')
    # test_tensor = torch.tensor(test_np, dtype=torch.float32)

    # Load trained models
    best_trained_model = SimpleGRU(8, 288, 32, 3).to('cuda')
    data_path = "best_model.pkl"
    with open(data_path, "rb") as fp:
        best_checkpoint_data = pickle.load(fp)
    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    predictions, actual = make_predictions(best_trained_model, test_tensor, device="cuda")
    
    scores = evaluate_metrics(actual, predictions)
    print(f'Predictions: MAE {scores[0]}, MAPE {scores[1]}, R^2 {scores[2]}')

    time_interval = pd.date_range(start="2022-01-02", periods=len(actual), freq='5min')
    plt.figure(figsize=(40,10))
    plt.plot(time_interval, actual, label='Actual')
    plt.plot(time_interval, predictions, label='Predicted')
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Load', fontsize=15)
    plt.title('Actual & Predicted Loads over Time', fontsize=20)
    plt.savefig('best_model.png')
    plt.show()