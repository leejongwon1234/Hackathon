import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Dropout

# Define a function to process the data and train the model
def process_and_train(model, dataset, scaler=None, initial=False):
    data = pd.read_csv(dataset)
    X = data[features]
    y = data['uenomax']
    
    # Standardize the data
    if initial:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    # Convert data to 3D for LSTM
    X_3d, y_3d = create_dataset(X_scaled, y, time_steps)

    # Train the model
    model.fit(X_3d, y_3d, epochs=30, batch_size=16, verbose=1)

    return model, scaler

# Convert data to 3D for LSTM
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps + 1):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps - 1])
    return np.array(Xs), np.array(ys)

# Define a function to predict values for a dataset
def predict_for_dataset(model, scaler, dataset_name):
    data = pd.read_csv(dataset_name)
    add_data = data[144:576].copy()
    data = pd.concat([add_data, data], ignore_index=True)
    X = data[features]
    X_scaled = scaler.transform(X)
    X_3d, _ = create_dataset(X_scaled, np.zeros(len(X_scaled)), time_steps)
    predictions = model.predict(X_3d)
    
    return predictions

# Features list
features = ['erabaddatt', 'erabaddsucc', 'endcmodbysgnbatt', 'endcmodbysgnbsucc', 'nummsg3', 'endcaddatt', 'endcaddsucc', 'connestabatt', 'connestabsucc', 'endcmodbymenbatt', 'endcmodbymenbsucc', 'numrar', 'rachpreamblea', 'handoveratt', 'handoversucc', 'endcrelbymenb', 'totprbulavg', 'airmaculbyte', 'redirectiontolte_coverageout', 'totprbdlavg', 'rlculbyte']

time_steps = 432 #432 # 이틀

# Build the LSTM model
model = Sequential([
    LSTM(168, input_shape=(time_steps, len(features))),
    Dense(32, activation='tanh'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model using datasets in sequence
datasets = []
for alphabet in ['A','E','D','C','G','H','I']:
    datasets.append(f'./{alphabet}/{alphabet}_data.csv')

initial = True
scaler = None
for dataset in datasets:
    print(f"Training on {dataset}...")
    model, scaler = process_and_train(model, dataset, scaler, initial)
    initial = False

# Predict for B_data.csv and J_data.csv
b_predictions = predict_for_dataset(model, scaler, './B/B_data.csv')
j_predictions = predict_for_dataset(model, scaler, './J/J_data.csv')

#한개씩 당기기 ##################################################################
b_predictions = b_predictions[1:]
j_predictions = j_predictions[1:]

print("Predictions for B_data.csv:", b_predictions)
print("Predictions for J_data.csv:", j_predictions)


# Convert negative predictions to 0
b_predictions[b_predictions < 0] = 0
j_predictions[j_predictions < 0] = 0

# Load B_data and J_data
B_data = pd.read_csv('./B/B_data.csv')
J_data = pd.read_csv('./J/J_data.csv')

# Add the predictions to B_data and J_data
B_data['uenomax'] = b_predictions
J_data['uenomax'] = j_predictions

# Save the updated datasets
B_data.to_csv('B_data_with_predictions.csv', index=False)
J_data.to_csv('J_data_with_predictions.csv', index=False)

test_df = pd.read_csv("Q1_label_sample.csv")
input_b_df = pd.read_csv("B_data_with_predictions.csv")
input_j_df = pd.read_csv("J_data_with_predictions.csv")
dates_to_remove = ['2023-06-12 18:15:00', '2023-07-05 04:30:00']
input_b_df = input_b_df[~input_b_df['datetime'].isin(dates_to_remove)]
input_j_df = input_j_df[~input_j_df['datetime'].isin(dates_to_remove)]

#index 삭제
input_b_df = input_b_df.reset_index(drop=True)
input_j_df = input_j_df.reset_index(drop=True)

for i in range(len(input_b_df)):
    test_df['BaseStationJ'][i] = input_j_df['uenomax'][i]
    test_df['BaseStationB'][i] = input_b_df['uenomax'][i]

test_df.to_csv("output_5M_DM_432TS_30E_16BS_FX_20F.csv", index=False)
