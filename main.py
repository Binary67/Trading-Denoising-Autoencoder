import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import warnings
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy

warnings.filterwarnings("ignore")

sys.path.append('C:/Users/frank/Desktop/Data Science/Projects/TradingAI')

from Utils.DataProcessing import TradingDataAggregator
from NoiseGenerator import *
from DenoisingAutoencoder import Conv1DAutoencoderDenoising
from Utils.BacktestingMod import BackTestStrategy
from SimpleMA import GenerateSmaCrossoverSignal

os.chdir('C:/Users/frank/Desktop/Data Science/Projects/TradingAI/Strategies/DenoisingTechniques')

ModelsPath = "ModelsPath/Autoencoder.pth"
ResumeTraining = False
PlotDiagram = True

# Hyperparameters
SequenceLength = 64
NumFeatures = 5

if __name__ == "__main__":

    ########################
    ### Data Preparation ###
    ########################
    Data = pd.read_csv('C:/Users/frank/Desktop/Data Science/Projects/TradingAI/Data/BTC-USD_1H(2017-2023).csv')
    Data = Data.tail(5000000)
    Data = Data.rename(columns={"timestamp": "Datetime", 
                                'open': 'Open', 
                                'high': 'High', 
                                'low': 'Low', 
                                'close': 'Close', 
                                'volume': 'Volume'})
    Data = Data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
    Data['Datetime'] = pd.to_datetime(Data['Datetime'])
    Data = Data.sort_values(by = 'Datetime', ascending = True)

    ######################
    ### Aggregate Data ###
    ######################

    AggregatorObj = TradingDataAggregator(Data, Interval = 60)
    Data = AggregatorObj.AggregateData()

    # Split Data
    Data = Data.iloc[:-1000]
    BacktestData = Data.iloc[-1000:]

    ######################
    ### Generate Noise ###
    ######################

    SmoothingParams = {"Span": 5}

    DataNoisy, DataOriginal, Scaler = GenerateNoisyTradingData(
        OriginalData = Data,
        SmoothingMethod = "EMA",
        SmoothingParams = SmoothingParams,
        NoiseMethod = "SimpleSampling",
        BlockSize = 5,
        ApplyScaling = False
    )

    #####################
    ### Data Training ###
    #####################

    Autoencoder = Conv1DAutoencoderDenoising()
    Autoencoder.LoadData(DataNoisy, DataOriginal)

    TrainNoisyTensor = Autoencoder.TransformData(Autoencoder.NoisyData)
    TrainCleanTensor = Autoencoder.TransformData(Autoencoder.CleanData)

    ValNoisyTensor = TrainNoisyTensor
    ValCleanTensor = TrainCleanTensor

    if os.path.exists(ModelsPath):
        print('Loading Existing Model')
        Autoencoder.BuildModel(SequenceLength, NumFeatures)
        Autoencoder.LoadModel(ModelsPath, ResumeTraining = True)
        if ResumeTraining:
            Autoencoder.TrainModel(TrainNoisyTensor, TrainCleanTensor, ValNoisyTensor, ValCleanTensor, Epochs = 500)
            Autoencoder.SaveModel(ModelsPath)
    else:
        print('No Model Found. Training New Model')
        Autoencoder.BuildModel(SequenceLength, NumFeatures)
        Autoencoder.TrainModel(TrainNoisyTensor, TrainCleanTensor, ValNoisyTensor, ValCleanTensor, Epochs = 500)
        Autoencoder.SaveModel(ModelsPath)

    Metrics = Autoencoder.EvaluateModel(ValNoisyTensor, ValCleanTensor)

    # Make Predictions
    DenoisedOutput = Autoencoder.Predict(BacktestData)

    ####################
    ### Plot Diagram ###
    ####################
    if PlotDiagram:
        BacktestData_Close = BacktestData["Close"].tolist()
        DenoisedOutput_Close = DenoisedOutput["Close"].tolist()

        plt.figure(figsize=(10, 6))

        plt.plot(BacktestData_Close, label="Origianl Data")
        plt.plot(DenoisedOutput_Close, label="Denoised Data")

        plt.title("Closed Values Comparison")
        plt.xlabel("Index")
        plt.ylabel("Closed")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    ###################
    ### Backtesting ###
    ###################
    DenoiseData = GenerateSmaCrossoverSignal(DenoisedOutput, ShortWindow = 50, LongWindow = 200)
    OriginalData = GenerateSmaCrossoverSignal(BacktestData, ShortWindow = 50, LongWindow = 200)

    DenoisedBacktestObj = Backtest(DenoiseData, BackTestStrategy, cash = 200000, commission = 0.0025)
    DenoiseStats = DenoisedBacktestObj.run()
    print(DenoiseStats)

    OriginalBacktestObj = Backtest(OriginalData, BackTestStrategy, cash = 200000, commission = 0.0025)
    OriginalStats = OriginalBacktestObj.run()
    print(OriginalStats)

    print()