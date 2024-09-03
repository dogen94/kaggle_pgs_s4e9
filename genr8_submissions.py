
# Standard Library imports
import os

# Third Party imports
import pandas as pd
# Evaluate the model
from sklearn.metrics import mean_squared_error

# Local imports
import models
import data


# Get dir to this script
DIRPATH = os.path.dirname(os.path.abspath(__file__))
# Model performance history file name
f_mhist = "submissions/mhist.txt"
# Encode cols
ENCODE_COLS = ["brand", "model", "fuel_type", "transmission",
               "ext_col", "int_col", "accident", "clean_title"]

# Genr8 gbtm
def genr8_gbtm():
    # Load training data
    dpath = os.path.join(DIRPATH, "data/train.csv")
    data_pd = data.load_csv(dpath)
    # Pop id col
    _ = data_pd.pop("id")
    # Ignore engine for now
    data_pd.pop("engine")
    # Separate data
    X = data_pd.drop('price', axis=1)
    # Pop prices
    prices = data_pd["price"]
    # Generate model name
    mname = models.genr8_model_name("gbtm")
    # Save model to this path
    mout = os.path.join(DIRPATH, f"trained_models/{mname}.pck")
    # Train gbtm
    model = models.train_gbtm(X, prices, fname=mout)
    # Read in test data
    tdpath = os.path.join(DIRPATH, "data/test.csv")
    tdata_pd = data.load_csv(tdpath)
    test_ids = tdata_pd.pop("id")
    # Ignore engine for now
    tdata_pd.pop("engine")
    # Separate data
    X_test = tdata_pd.drop('price', axis=1)
    # Pop prices
    prices_test = tdata_pd["price"]
    # Test performance
    prices_pred = model.predict(X_test)
    mse = mean_squared_error(prices_test, prices_pred)
    # Save this to history
    mhist_path = os.path.join(DIRPATH, f_mhist)
    # Record model performance
    with open(mhist_path, 'a') as file:
        # Construct output
        oline = f"{mname},{mse}\n"
        file.write(oline)

genr8_gbtm()