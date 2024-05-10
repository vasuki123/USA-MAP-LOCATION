import math

def feature_scaling(data):
    scaled_data = []
    for row in data:
        scaled_row = [math.sqrt(feature) for feature in row]
        scaled_data.append(scaled_row)
    return scaled_data

def train_test_split(data, target, test_size=0.2):
    split_index = math.ceil(len(data) * (1 - test_size))
    X_train, X_test = data[:split_index], data[split_index:]
    y_train, y_test = target[:split_index], target[split_index:]
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    X_train_T = [list(t) for t in zip(*X_train)]
    XTX = [[sum(X_train_T[i][j] * X_train_T[i][k] for i in range(len(X_train_T))) for k in range(len(X_train_T))] for j in range(len(X_train_T))]
    XTX_inv = [[1.0 if i == j else 0.0 for i in range(len(X_train_T))] for j in range(len(X_train_T))]
    for i in range(len(XTX)):
        for j in range(len(XTX[0])):
            XTX_inv[i][j] -= XTX[i][j]
    model_params = [sum(XTX_inv[i][j] * sum(X_train_T[j][k] * y_train[k] for k in range(len(X_train_T))) for j in range(len(X_train_T))) for i in range(len(X_train_T))]
    return model_params

def predict(model_params, X_test):
    predictions = [sum(model_params[i] * X_test[j][i] for i in range(len(X_test[0]))) for j in range(len(X_test))]
    return predictions

def evaluate(y_true, y_pred):
    mse = sum(math.pow(y_true[i] - y_pred[i], 2) for i in range(len(y_true))) / len(y_true)
    rmse = math.sqrt(mse)
    return rmse

def mean(numbers):
    if not numbers:
        raise ValueError("List of numbers cannot be empty")
    return sum(numbers) / len(numbers)

def median(numbers):
    if not numbers:
        raise ValueError("List of numbers cannot be empty")
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 0:
        return (sorted_numbers[n // 2 - 1] + sorted_numbers[n // 2]) / 2
    else:
        return sorted_numbers[n // 2]

def count(numbers):
    return len(numbers)
