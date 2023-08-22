import pandas as pd

new_train_data = pd.read_csv('읽을 데이터 이름')

correlation_train = new_train_data.corr()['상관관계를 관찰하고 싶은 열이름'].drop('상관관계를 관찰하고 싶은 열이름')

# 내림차순으로 정렬
correlation_train = correlation_train.sort_values(ascending=False)

#correlation_train의 이름을 리스트로 변환
correlation_train = correlation_train.index.tolist()
print(correlation_train)
