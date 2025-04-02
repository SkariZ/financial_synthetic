# Stock Market Prediction: Analyzing Trends and Forecasting Future Movements 
This repository contains data and information of the financial project in course TIF360/FYM360.

## Motivation
Predicting the stock market is a complex challenge that involves analyzing historical data to uncover patterns and trends that may offer insights into future movements. While some argue that market prediction is futile due to inherent uncertainties, advances in deep learning techniques offer potential strategies for gaining an edge.

This project explores predictive techniques to forecast future movements and make informed weighting decisions in a stock portfolio.

## Research Objective
Develop deep learning models capable of forecasting future trends in synthetic financial data and formulate asset allocation strategies optimized based on these predictions and/or with other methods.

## Dataset
This project contains a few datasets to investigate.

1. Synthetic data of **20** geometric brownian motions (GBMs) of time series with **shared** drift and volatility parameter:

<p align="center">
  <img src="Imgs/simulated_series_plot.png" alt="Simulated stocks with same drift and volatility" width="700">
</p>

2. Synthetic data of **20** geometric brownian motions (GBMs) of time series with **independent** drift and volatility parameter butwith **shared** seasonality:

<p align="center">
  <img src="Imgs/simulated_series_seasonality_plot.png" alt="Simulated stocks with same drift and volatility" width="700">
</p>

3. A simulated portfolio of **180** stocks from different distributions with some shared characteristics, and some not:

<p align="center">
  <img src="Imgs/simulated_stock_prices.png" alt="Simulated stocks with same drift and volatility" width="700">
</p>

## Installation
To run this project, ensure you have the following dependencies installed:
```bash
pip install numpy pandas matplotlib scipy
```




