import matplotlib.pyplot as plt
import pandas as pd

####################################

def plotFeature_byTime(data:pd.DataFrame, feature:str, time:str = "elapsed_s"):
    """
    Plots given feature as time passes, using matplotlib.plot()
    """
    plt.figure(figsize=(20,5))
    plt.plot(data[time.lower()], data[feature.lower()])
    plt.title(f"{feature.lower()}'s change as time goes by")
    plt.ylabel(f"{feature.lower()}")
    plt.xlabel("Time (s)")
    plt.show()
    