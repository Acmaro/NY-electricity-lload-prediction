import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from training import SimpleLSTM, SimpleGRU

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
        A tensor of shape [N, 7] containing continuous test data for 2020.
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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Parameters for data preprocessing
    input_seq_len = 288
    output_seq_len = 288
    batch_size = 128
    # Parameters for model initialization and training
    num_features = 8
    hidden_size = 32
    num_layers = 2

    lstm_model = SimpleLSTM(num_features, output_seq_len, hidden_size, num_layers).to(device)
    lstm_model.load_state_dict(torch.load('LSTM.pt', weights_only=True))

    gru_model = SimpleGRU(num_features, output_seq_len, hidden_size, num_layers).to(device)
    gru_model.load_state_dict(torch.load('GRU.pt', weights_only=True))

    # Load test data
    test_np = np.load('test_data.npy')
    test_tensor = torch.tensor(test_np, dtype=torch.float32)

    predictions_lstm, actual = make_predictions(lstm_model, test_tensor, device=device)
    predictions_gru, _ = make_predictions(gru_model, test_tensor, device=device)

    lstm_scores = evaluate_metrics(actual, predictions_lstm)
    print(f'LSTM: MAE {lstm_scores[0]}, MAPE {lstm_scores[1]}, R^2 {lstm_scores[2]}')
    gru_scores = evaluate_metrics(actual, predictions_gru)
    print(f'GRU: MAE {gru_scores[0]}, MAPE {gru_scores[1]}, R^2 {gru_scores[2]}')

    font = {'family' : 'monospace',
            'weight' : 'normal',
            'size'   : 22}

    plt.rc('font', **font)

    time_interval = pd.date_range(start="2020-01-02", periods=len(actual), freq='5min')
    plt.figure(figsize=(40,10))
    plt.plot(time_interval, actual, label='Actual')
    # plt.plot(time_interval, predictions_rnn, label='Predicted (RNN)')
    plt.plot(time_interval, predictions_lstm, label='Predicted (LSTM)')
    plt.plot(time_interval, predictions_gru, label='Predicted (GRU)')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Load')
    plt.title('Actual & Predicted Loads over Time')
    plt.show()