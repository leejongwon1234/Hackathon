import pandas as pd

new_train_data = pd.read_csv('Q1_train.csv')

correlation_train = new_train_data.corr()['uenomax'].drop('uenomax')

# 내림차순으로 정렬
correlation_train = correlation_train.sort_values(ascending=False)

#correlation_train의 이름을 리스트로 변환
correlation_train = correlation_train.index.tolist()
print(correlation_train)