import pandas as pd
import matplotlib.pyplot as plt

# Load the train data
train_data = pd.read_csv('데이터 시각화할 CSV파일 이름')
# Convert the 'datetime' column to datetime format for the train data
train_data['datetime'] = pd.to_datetime(train_data['datetime'])

# Load the test data
test_data = pd.read_csv('데이터 시각화할 CSV파일 이름 2')
# Convert the 'datetime' column to datetime format for the test data
test_data['datetime'] = pd.to_datetime(test_data['datetime'])

# Combine the train and test data
combined_data = pd.concat([train_data, test_data], ignore_index=True)

# Define a list of distinct colors for better visualization
distinct_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FFA500']

# Resample and plot function
def resample_and_plot(column_name, data_to_use):
    daily_avg = data_to_use.groupby('그룹화할 열 이름').resample('D', on='datetime').mean().reset_index()
    
    plt.figure(figsize=(10, 10))
    for idx, ru_id in enumerate(daily_avg['그룹화할 열 이름'].unique()):
        subset = daily_avg[daily_avg['그룹화할 열 이름'] == ru_id]
        plt.plot(subset['datetime'], subset[column_name], label=ru_id, color=distinct_colors[idx % len(distinct_colors)])
    
    plt.title(f'Daily Average {column_name.upper()} by RU_ID')
    plt.xlabel('Date')
    plt.ylabel(f'Average {column_name.upper()}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

resample_and_plot('데이터 시각화할 열 이름', train_data)

for column_name in ['데이터 시각화할 열 이름 리스트']:

    resample_and_plot(column_name, combined_data)

# Resample and plot the 'erabaddatt' for the combined data
#resample_and_plot('erabaddatt', combined_data)
