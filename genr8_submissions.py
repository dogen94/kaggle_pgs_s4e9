
# Standard Library imports
import os

# Third Party imports
import pandas as pd
import numpy as np

# Evaluate the model
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
SCALE_COLS = ["model_year", "milage"]

# Genr8 gbtm
def genr8_gbtm():
    # Load training data
    dpath = os.path.join(DIRPATH, "data/train.csv")
    data_pd = data.load_csv(dpath)
    # Pop id col
    _ = data_pd.pop("id")
    # Ignore engine for now
    data_pd.pop("engine")
    # Pop prices
    prices = data_pd.pop("price")
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ENCODE_COLS),
            ('num', 'passthrough', SCALE_COLS)
        ])
    # Create a pipeline that preprocesses the data
    pp_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    pp_pipeline.fit(data_pd)
    # Preprocess data
    data_pd_pp = pp_pipeline.transform(data_pd)
    data_pp = data_pd_pp.toarray()
    # Build df from transformed data
    feature_names = pp_pipeline.named_steps['preprocessor'].get_feature_names_out()
    data_pp_df = pd.DataFrame(data_pp, columns=feature_names)
    # Generate model name
    mname = models.genr8_model_name("gbtm")
    # Save model to this path
    mout = os.path.join(DIRPATH, f"trained_models/{mname}.pck")
    # Train gbtm
    model = models.train_gbtm(data_pp_df, prices, fname=mout)
    # Read in test data
    tdpath = os.path.join(DIRPATH, "data/test.csv")
    tdata_pd = data.load_csv(tdpath)
    test_ids = tdata_pd.pop("id")
    # Ignore engine for now
    tdata_pd.pop("engine")
    # # Separate data
    # X_test = tdata_pd_pp.drop('price', axis=1)
    # prices_test = tdata_pd_pp.pop("price")
    # Preprocess test data
    tdata_pd_pp = pp_pipeline.transform(tdata_pd)
    test_data_pp = tdata_pd_pp.toarray()
    # Build df from transformed data
    feature_names = pp_pipeline.named_steps['preprocessor'].get_feature_names_out()
    tdata_pp_df = pd.DataFrame(test_data_pp, columns=feature_names)
    # Test performance
    prices_pred = model.predict(tdata_pp_df)
    # mse = mean_squared_error(prices_test, prices_pred)
    submission = np.vstack([test_ids.to_numpy(), prices_pred])
    submission_pd = pd.DataFrame(np.vstack([test_ids.to_numpy(), abs(prices_pred)]).T,
                                 columns=["id", "price"])
    submission_pd = submission_pd.convert_dtypes()
    submission_pd.to_csv(os.path.join(DIRPATH, f"submissions/{mname}.csv"), index=False)
    # Save this to history
    # mhist_path = os.path.join(DIRPATH, f_mhist)
    # # Record model performance
    # with open(mhist_path, 'a') as file:
    #     # Construct output
    #     oline = f"{mname},{mse}\n"
    #     file.write(oline)


genr8_gbtm()