import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indicies = shuffled[:test_set_size]
    train_indicies = shuffled[test_set_size:]
    return data.iloc[train_indicies], data.iloc[test_indicies]

if __name__ == '__main__':

    df = pd.read_csv('data.csv')
    train, test = data_split(df, 0.3)

    x_train = train[['fever', 'bodypain', 'age', 'RunnyNose', 'Breath-Difficulty']].to_numpy()
    x_test = test[['fever', 'bodypain', 'age', 'RunnyNose', 'Breath-Difficulty']].to_numpy()

    y_train = train[['CoronaVirus-Prob']].to_numpy().reshape(2902, )
    y_test = test[['CoronaVirus-Prob']].to_numpy().reshape(1243, )

    lg = LogisticRegression()
    lg.fit(x_train, y_train)

    file = open('model.pkl', 'wb')
    pickle.dump(lg, file)
    file.close()

    # input_features = [101.28, 0, 54, 1, -1]
    # result = lg.predict([input_features])
    # print(result)







