import numpy as np
from sklearn.model_selection import cross_val_score
import pandas as pd

#########################

def calc_rmse(model, validation, data:pd.DataFrame, features:list, y:list) -> float: 
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

def feature_selection(model, validation, data:pd.DataFrame, features:list, y:list):
    """
    Calculates the best features by minimising RMSE
    """
    def evaluate(feature_subset):
        return calc_rmse(model,validation,data,feature_subset,y)
    
    selected_features = list(features)
    full_rmse = evaluate(selected_features)

    results = [{
    "n_features": len(selected_features),
    "rmse": full_rmse,
    "features": selected_features.copy()
    }]

    while len(selected_features) > 1:
        candidates = []

        for feature in selected_features:
            remaining = [
                column for column in selected_features
                if column != feature
            ]

            candidates.append({
                "n_features": len(remaining),
                "rmse": evaluate(remaining),
                "features": remaining
            })

        best_candidate = min(candidates, key=lambda result: result["rmse"])

        selected_features = best_candidate["features"].copy()
        results.append(best_candidate)

    results_df = pd.DataFrame(results)
    
    print(
    results_df[["features", "rmse"]].to_string(
        index=False,
        formatters={"features": lambda names: ", ".join(names)}
    )
)

