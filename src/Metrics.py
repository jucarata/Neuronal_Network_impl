def acc(y_hat, y):
    correct_preds = sum(1 for pred, real in zip(y_hat, y) if pred == real)
    total = len(y)

    accuracy = correct_preds / total
    return accuracy