import numpy as np
import pandas as pd
import math
import json
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# My custom XGBoost wrapper, which is meant to be
# an interface so we can treat it like a
# forecasting model.
class XGBoostTimeSeries:
    
    # Constructor. Simply record the training data,
    # which is expected to be a pandas dataframe,
    # in chronological time order, with columns:
    # [`Close`]: silver closing price
    # [`High`]: the highest silver price on the given date
    # [`Low`]: the lowest silver price on the given date
    # [`Volume`]: the traded volume on the given date
    def __init__(self,train_data):
        self.train_data = train_data
        self.rolling = []
        self.state = []
    
    # Takes in a time series of rows
    # and applies our engineered features.
    def __f(self,table):
        table_copy = table.copy()
        
        # Closing price returns
        table_copy["Close_returns"] = (table["Close"] - table["Close"].shift(1))/(table["Close"].shift(1))
        
        # Intraday volatility measure
        table_copy["LowHigh_rel"] = (table_copy["High"] - table_copy["Low"]) / (table_copy["Low"])
        
        # 10-day rolling average returns ( < 10)
        table_copy["Close_returns_roll10mean"] = table_copy["Close_returns"].rolling(10).mean()
        
        # 10-day rolling returns std deviation (volatility measure)
        table_copy["Close_returns_roll10std"] = table_copy["Close_returns"].rolling(10).std()
        
        # Custom volume measure
        table_copy["Volume_rel"] = table_copy["Volume"] / (table_copy["Volume"].rolling(10).mean())
        
        return table_copy[["Close_returns", "Close_returns_roll10mean", "Close_returns_roll10std", "LowHigh_rel", "Volume_rel"]]
        
    
    # This fills in the supervised learning table!
    def fill_table(self):
        
        # Start with a copy of the original df, with only the relevant colums
        self.train_data = self.train_data[["Close", "High", "Low", "Volume"]]
        self.train_table = self.train_data
        
        # Fill in the engineered features
        self.train_table = self.__f(self.train_table)
        l = self.train_table
        # Create the target column
        self.train_table["Target"] = self.train_table["Close_returns"].shift(-1)
        
        # Keep only whats relevant for training
        self.train_table = self.train_table.dropna()
        
        # Finally, we'll store the last 11 values for later
        # forecasting
        self.last = self.train_data.tail(11)[["Close", "High", "Low", "Volume"]]
        return l
    
    # Method to update the stored values
    def update(self,new_sample):
        # Keep what's relevant
        new_sample = new_sample[["Close", "High", "Low", "Volume"]]
        # Remove the first row
        self.last = self.last.tail(len(self.last) - 1)
        # Append the new row
        self.last = pd.concat([self.last, new_sample], axis=0)
        
    # Forecast 1 step ahead
    def forecast(self):
        # First, compute the relevant features from the
        # previously stored 12 values
        engineered_features = self.__f(self.last)
        # Now produce the sample
        sample = engineered_features.tail(1)
        # Get the predicited return
        predicted_ret = self.model.predict(sample).item()
        # To return the forecasted closing price,
        # we need the last closing price, and make the
        # adjustment
        last_closing_price = self.last.tail(1)["Close"].item()
        return (last_closing_price + predicted_ret * last_closing_price)
    
    # Function to fit the final model, after optimising hyperparameters
    def fit(self):
        # Now we take the full training data
        X_train,y_train = self.train_table[["Close_returns", "Close_returns_roll10mean", "Close_returns_roll10std", "LowHigh_rel", "Volume_rel"]], self.train_table[["Target"]]
        self.model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=self.best_iteration,
            subsample=0.7,
            colsample_bytree=0.7,
            learning_rate=self.tuned_hyperparams["learning_rate"],
            max_depth=self.tuned_hyperparams["max_depth"],
            reg_lambda=self.tuned_hyperparams["reg_lambda"],
            min_child_weight=self.tuned_hyperparams["min_child_weight"]
        )
        self.model.fit(
            X_train, y_train,
            verbose=False
        )
    
    # Function to do a small grid search to find decent
    # XGBoost hyperparameters. We restrict to searching a small number
    # of learning rates, depths, and lambda (L2 regularisation param.)
    def tune(self,initial_size, validation_sets, hyperparams):
        
        # How much get's added each walk foward?
        validation_increment = math.floor((len(self.train_table) - initial_size)/(validation_sets))
        final_validation_set_size = len(self.train_table) - (initial_size + (validation_sets-1) * validation_increment)
        
        
        # Keep dictionaries with the rmse's obtained from each
        # hyperparameter combination, as well as the number of
        # boosted iterations where the best performance
        # was obtained on the validation set.
        rmse_dic = {}
        best_iteration_dic = {}
        
        # We start enumerating the walk forward
        for i in range(0,validation_sets):
            # Get the train-validation split sizes
            train_size = 0
            validation_size = 0
            if (i == validation_sets-1):
                validation_size = final_validation_set_size
                train_size = len(self.train_table) - validation_size
            else:
                validation_size = validation_increment
                train_size = initial_size + i * (validation_increment)
            train_df = self.train_table.head(train_size)
            validation_df = self.train_table.head(train_size + validation_size).tail(validation_size)
            
            # Now get the splits
            X_train,y_train = train_df[["Close_returns", "Close_returns_roll10mean", "Close_returns_roll10std", "LowHigh_rel", "Volume_rel"]], train_df[["Target"]]
            X_valid,y_valid = validation_df[["Close_returns", "Close_returns_roll10mean", "Close_returns_roll10std", "LowHigh_rel", "Volume_rel"]], validation_df[["Target"]]
            
            # Now enumerate over hyperparameters, fit, and record scores along with best_iteration
            for hp in hyperparams:
                model = XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=1000,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    learning_rate=hp["learning_rate"],
                    max_depth=hp["max_depth"],
                    eval_metric="rmse",
                    early_stopping_rounds=50,
                    reg_lambda=hp["reg_lambda"],
                    min_child_weight=hp["min_child_weight"],
                    tree_method="hist",
                    n_jobs=-1
                )

                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    verbose=False
                )

                rmse = model.evals_result()["validation_0"]["rmse"][model.best_iteration]
                best_iteration = model.best_iteration
                key = tuple(sorted(hp.items())) # We can't use a dictionary as a key as it's not immutable. Tuples are immutable.
                rmse_dic.setdefault(key, []).append(rmse)
                best_iteration_dic.setdefault(key,[]).append(best_iteration)
            
            # Store the mean rmse over the validation sets
            # as our final selection metric
            mean_rmse_dic = {}
            for key in rmse_dic:
                mean_rmse_dic[key] = np.mean(np.array(rmse_dic[key]))
            tuned_hyperparams = min(mean_rmse_dic,key=mean_rmse_dic.get)
            self.tuned_hyperparams = dict(tuned_hyperparams)
            # Now get the n_estimators we'll use for the final fit
            self.best_iteration = math.ceil(np.median(best_iteration_dic[key]))
        return self.tuned_hyperparams,self.best_iteration