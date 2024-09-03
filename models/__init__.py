import os
from datetime import datetime

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import pickle


# First try the naive GB tree model
def train_gbtm(X,y, fname=None):
    # Define the parameter grid
    # param_grid = {
    #     'n_estimators': [50, 100, 200],
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'max_depth': [3, 5, 7, 9],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4, 8]
    # }
    model = GradientBoostingRegressor()
    model.fit(X, y)
    # grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
    #                            cv=5, scoring='neg_mean_squared_error',
    #                            n_jobs=-1, verbose=1)
    # grid_search.fit(X, y)
    # best_model = grid_search.best_estimator_
    # Try saving model
    if fname:
        save_model(best_model, fname)
    # Return handle to best model
    return best_model


# Pickle model state
def save_model(model, fname, force=False):
    # Validate inputs
    if not fname:
        raise ValueError("File path cannot be empty.")
    if not model:
        raise ValueError("Model cannot be empty.")
    
    # Check if file exists and force flag is not set
    if os.path.exists(fname) and not force:
        print(f"The file {fname} already exists. Use 'force=True' to overwrite.")
        return

        # Save the provided model
    with open(fname, 'wb') as file:
        pickle.dump(model, file)


def genr8_model_name(suffix):
    # Get the current date and time
    now = datetime.now()
    # Format the date and time into a string for the file name
    time_stamp = now.strftime('%Y-%m-%d_%H-%M-%S')
    # Model name
    mname = f"{suffix}-{time_stamp}"
    return mname