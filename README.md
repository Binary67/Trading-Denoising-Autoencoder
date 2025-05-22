# Trading-Denoising-Autoencoder (Trading-DAE)

This project implements a 1D Convolutional Denoising Autoencoder (DAE) designed to reduce noise from financial time series data (OHLCV). The denoised data is then used with a Simple Moving Average (SMA) crossover strategy, and its performance is compared against the same strategy using the original, noisy data through backtesting.

The core idea is to train the DAE to distinguish signal from noise by feeding it synthetically generated noisy data and teaching it to reconstruct the original, cleaner version.

## Key Features

*   **1D Convolutional Denoising Autoencoder:** Employs a `Conv1D` based autoencoder architecture tailored for sequential trading data.
*   **Synthetic Noise Generation:** Includes a pipeline to create noisy training samples from clean data by:
    *   Smoothing the original data (EMA, SMA, SavGol).
    *   Calculating residuals (original - smoothed).
    *   Generating noise patterns from these residuals.
    *   Adding this noise back to the original data.
    *   Enforcing OHLCV constraints on the noisy data.
*   **Model Training & Management:**
    *   Trains the DAE using noisy input and clean target data.
    *   Supports saving the best performing model during training.
    *   Allows loading of pre-trained models for inference or further training.
*   **Data Transformation:** Preprocesses data into sequences suitable for the 1D CNN and applies Z-score normalization.
*   **Trading Strategy Integration:**
    *   Generates trading signals using an SMA crossover strategy (`SimpleMA.py`).
    *   Applies this strategy to both the original and the DAE-denoised data.
*   **Backtesting:**
    *   Utilizes the `backtesting.py` library to evaluate and compare the performance of the trading strategy on original vs. denoised data.
*   **Visualization:** Plots a comparison of original and denoised 'Close' prices to visually assess the DAE's impact.
