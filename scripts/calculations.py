import numpy as np
from sklearn.model_selection import cross_val_score

#########################

def calc_rmse(model, validation, data:list, features:list, y:list) -> float: 
    """
    Calculates RMSE using crossvalidation score from sklearn
    """
    scores = cross_val_score(
            model,
            data[features],
            y,
            cv=validation,
            scoring="neg_root_mean_squared_error"
        )
    return round(-scores.mean(),3)