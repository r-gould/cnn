import time

class Display:
    def __init__(self):
        self.delta_time = 0

    @staticmethod
    def display_train_statistics(epoch, train_time, predict_time, train_stats=None, valid_stats=None):
        message = f"""For epoch {epoch}:
    """
        if train_stats:
            (train_cost, train_acc) = train_stats
            train_cost = round(train_cost, 2)
            train_acc = round(train_acc * 100, 3)

            message += f"""Training cost: {train_cost}
    Training accuracy: {train_acc}%
    
    """
        
        if valid_stats:
            (valid_cost, valid_acc) = valid_stats
            valid_cost = round(valid_cost, 2)
            valid_acc = round(valid_acc * 100, 3)

            message += f"""Validation cost: {valid_cost}
    Validation accuracy: {valid_acc}%
    
    """

        train_time = round(train_time, 2)
        predict_time = round(predict_time, 2)

        message += f"""Time to train: {train_time}s
    Time to predict: {predict_time}s
    """

        print(message)

    @staticmethod
    def display_evaluation_statistics(acc):
        acc = round(acc * 100, 3)
        print(f"Prediction accuracy: {acc}%\n")
