import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from scripts.calculations import feature_selection

#################################
def generate_df(dataframe:pd.DataFrame):
    """
    Makes dataframe with the features needed for task 0. 
    Keep in mind, this function is specifically coded for task 0!
    """
    n = dataframe["speed_rpm"] / 1000
    Q = dataframe["flow_m3_h"]
    y = dataframe["power_w"]

    X = pd.DataFrame(index=dataframe.index)
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 0:
                continue # this is the intercept
            X[f"n{i}_Q{j}"] = (n**i)*(Q**j)
    return X, y 

def main():
    df_train = pd.read_csv("data/case1/train1.csv")
    X_train, y_train = generate_df(df_train)

    model = LinearRegression()
    cros_val = KFold(n_splits=5, shuffle=True, random_state=42)
    features = list(X_train.columns)

    results = feature_selection(model,cros_val,X_train,features,y_train)
    best_result = results.loc[results["rmse"].idxmin()]

    print("Best result details:")
    print(best_result)

    
#################################
if __name__ == "__main__":
    main()