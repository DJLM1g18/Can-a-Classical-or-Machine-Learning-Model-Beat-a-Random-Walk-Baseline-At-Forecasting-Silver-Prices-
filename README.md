# Overview of the task

We take silver pricing data over a 10 year period and evaluate whether an ARIMA or XGBoost model are able to beat a random walk baseline model at forecasting the closing prices.

# Imports

If you wish to run this notebook, you'll need the following Python packages installed:

1. `numpy`
2. `pandas`
3. `matplotlib`
4. `statsmodels`
5. `dieboldmariano`.

Otherwise, you can simply download the `analysis_final.html` file and view the final report.

# ARIMA(p,d,q)

In order to find suitable parameters ${p,d,q}$ for our ARIMA model, we follow the Box-Jenkins methodology. This involves first identifying the smallest value of $d$ for which the ${d^{\text{th}}}$ differenced time series exhibits behaviour consistent with (weak) stationarity. In our case, we get ${d=1}$. Next, we identify a small region of candidate $p$ and $q$ values by plotting the ACF and PACF respectively. We note that the ACF and PACF die rapidly to 0 after lag 0, and thus check the grid ${p,q \{0,1,2\}}$. Finally, we use the AIC to select an ARIMA model from our candidates to bring forward for out-of-sample testing. ARIMA(0,1,2) achieves the lowest AIC, and so is the model we bring forward.

# XGBoost

We then applied XGBoost, a general-purpose supervised learning algorithm rather than a dedicated time‑series model. To use it for forecasting, we constructed temporal features that encode historical information. We also built a wrapper, XGBoostTimeSeries, which allows XGBoost to behave like a time‑series model and provides a consistent interface. Our implementation includes a tuning routine that performs walk‑forward validation combined with a small grid search to select hyperparameters before fitting the final model.

# Forecasting performance comparison

Finally, we compared the out‑of‑sample forecasting performance of all three models. The ARIMA model achieved the lowest RMSE, followed by XGBoost, with the random‑walk baseline performing worst. However, subsequent Diebold–Mariano tests indicated that none of the differences in forecasting accuracy were statistically significant, implying that all three models performed similarly from a statistical standpoint.
