import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

def SmoothData(DataFrame, SmoothingMethod="EMA", SmoothingParams=None):
    """
    Smooth each of the OHLCV columns using the specified smoothing method.
    
    Parameters:
        DataFrame (pd.DataFrame): Original trading data.
        SmoothingMethod (str): "EMA", "SMA", or "SavGol".
        SmoothingParams (dict): Parameters for the smoothing method.
            For EMA: {"Span": int} (default 5)
            For SMA: {"Window": int} (default 5)
            For SavGol: {"WindowLength": int, "PolyOrder": int} (e.g., 5 and 2)
    
    Returns:
        pd.DataFrame: Smoothed data with the same columns as the input.
    """
    # Set default parameters if none provided
    if SmoothingParams is None:
        if SmoothingMethod.upper() == "EMA":
            SmoothingParams = {"Span": 5}
        elif SmoothingMethod.upper() == "SMA":
            SmoothingParams = {"Window": 5}
        elif SmoothingMethod.upper() == "SAVGOL":
            SmoothingParams = {"WindowLength": 5, "PolyOrder": 2}
    
    # We assume that the first column (e.g., "Datetime") is not to be smoothed.
    ColumnsToSmooth = ["Open", "High", "Low", "Close", "Volume"]
    SmoothedData = DataFrame.copy()

    if SmoothingMethod.upper() == "EMA":
        Span = SmoothingParams.get("Span", 5)
        for Column in ColumnsToSmooth:
            # Use adjust=True to handle early values
            SmoothedData[Column] = DataFrame[Column].ewm(span=Span, adjust=True).mean()
            
    elif SmoothingMethod.upper() == "SMA":
        Window = SmoothingParams.get("Window", 5)
        for Column in ColumnsToSmooth:
            # Use min_periods=1 so that the beginning is handled properly
            SmoothedData[Column] = DataFrame[Column].rolling(window=Window, min_periods=1).mean()
            
    elif SmoothingMethod.upper() == "SAVGOL":
        WindowLength = SmoothingParams.get("WindowLength", 5)
        PolyOrder = SmoothingParams.get("PolyOrder", 2)
        for Column in ColumnsToSmooth:
            # savgol_filter requires odd window_length and at least polyorder+2 points.
            # Adjust window length if necessary.
            DataArray = DataFrame[Column].values
            if WindowLength % 2 == 0:
                WindowLength += 1
            if len(DataArray) < WindowLength:
                WindowLength = len(DataArray) if len(DataArray) % 2 != 0 else len(DataArray) - 1
            # Apply the Savitzky-Golay filter
            SmoothedData[Column] = savgol_filter(DataArray, window_length=WindowLength, polyorder=PolyOrder, mode='interp')
    
    return SmoothedData

def CalculateResiduals(OriginalData, SmoothedData):
    """
    Compute residuals (the high-frequency component) as the difference between the original data and the smoothed data.
    
    Parameters:
        OriginalData (pd.DataFrame): The original OHLCV data.
        SmoothedData (pd.DataFrame): The smoothed OHLCV data.
    
    Returns:
        pd.DataFrame: Residuals for each OHLCV column.
    """
    ColumnsToProcess = ["Open", "High", "Low", "Close", "Volume"]
    ResidualsData = OriginalData[ColumnsToProcess] - SmoothedData[ColumnsToProcess]
    return ResidualsData

def GenerateNoise(ResidualsData, NoiseMethod="SimpleSampling", BlockSize=5):
    """
    Generate noise by randomly sampling residuals for each column.
    Currently implements simple sampling.
    
    Parameters:
        ResidualsData (pd.DataFrame): The residuals calculated from the original data.
        NoiseMethod (str): "SimpleSampling" or "BlockSampling". (BlockSampling is not implemented here.)
        BlockSize (int): Size of the block for block sampling (if applicable).
    
    Returns:
        pd.DataFrame: Noise data with the same shape as the residuals.
    """
    NoiseData = pd.DataFrame(index=ResidualsData.index, columns=ResidualsData.columns)
    if NoiseMethod == "SimpleSampling":
        # For each column, sample randomly (with replacement) from the pool of non-null residual values.
        for Column in ResidualsData.columns:
            ResidualPool = ResidualsData[Column].dropna().values
            if len(ResidualPool) == 0:
                # If there are no residuals, use zeros.
                NoiseData[Column] = 0
            else:
                SampledNoise = np.random.choice(ResidualPool, size=len(ResidualsData), replace=True)
                NoiseData[Column] = SampledNoise
    else:
        raise NotImplementedError("BlockSampling is not implemented in this example.")
    return NoiseData

def AddNoise(OriginalData, NoiseData):
    """
    Add generated noise to the original data.
    
    Parameters:
        OriginalData (pd.DataFrame): Original OHLCV data.
        NoiseData (pd.DataFrame): Generated noise data.
    
    Returns:
        pd.DataFrame: Intermediate noisy data.
    """
    # Only add noise to OHLCV columns; assume the first column (e.g., Datetime) remains unchanged.
    ColumnsToProcess = ["Open", "High", "Low", "Close", "Volume"]
    NoisyData = OriginalData.copy()
    NoisyData[ColumnsToProcess] = OriginalData[ColumnsToProcess] + NoiseData[ColumnsToProcess]
    return NoisyData

def EnforceOHLCVConstraints(NoisyData):
    """
    Adjust the noisy data to enforce logical OHLCV constraints:
      - Volume must be non-negative.
      - High must be the maximum among Open, High, and Close.
      - Low must be the minimum among Open, Low, and Close.
      - Open and Close must lie between Low and High.
    
    Parameters:
        NoisyData (pd.DataFrame): Noisy OHLCV data before constraints.
    
    Returns:
        pd.DataFrame: Noisy data adjusted to satisfy constraints.
    """
    AdjustedData = NoisyData.copy()
    for Index, Row in AdjustedData.iterrows():
        # Extract intermediate values for OHLCV
        OpenValue = Row["Open"]
        HighValue = Row["High"]
        LowValue = Row["Low"]
        CloseValue = Row["Close"]
        VolumeValue = Row["Volume"]
        
        # Enforce Volume constraint
        AdjustedVolume = max(0, VolumeValue)
        
        # Adjust High and Low first using intermediate values
        AdjustedHigh = max(HighValue, OpenValue, CloseValue)
        AdjustedLow = min(LowValue, OpenValue, CloseValue)
        
        # Enforce Open and Close to lie within [AdjustedLow, AdjustedHigh]
        AdjustedOpen = max(AdjustedLow, min(AdjustedHigh, OpenValue))
        AdjustedClose = max(AdjustedLow, min(AdjustedHigh, CloseValue))
        
        # Update the row in the DataFrame
        AdjustedData.at[Index, "Open"] = AdjustedOpen
        AdjustedData.at[Index, "High"] = AdjustedHigh
        AdjustedData.at[Index, "Low"] = AdjustedLow
        AdjustedData.at[Index, "Close"] = AdjustedClose
        AdjustedData.at[Index, "Volume"] = AdjustedVolume
    
    return AdjustedData

def ScaleData(TrainData, TestData):
    """
    Fit a StandardScaler on TrainData and transform both TrainData and TestData.
    
    Parameters:
        TrainData (pd.DataFrame): Data to fit the scaler.
        TestData (pd.DataFrame): Data to be transformed using the fitted scaler.
    
    Returns:
        tuple: (ScaledTrainData, ScaledTestData, FittedScaler)
    """
    Scaler = StandardScaler()
    # Fit on OHLCV columns only; assuming the first column (e.g., Datetime) is non-numeric.
    ColumnsToScale = ["Open", "High", "Low", "Close", "Volume"]
    FittedScaler = Scaler.fit(TrainData[ColumnsToScale])
    ScaledTrainData = TrainData.copy()
    ScaledTestData = TestData.copy()
    ScaledTrainData[ColumnsToScale] = FittedScaler.transform(TrainData[ColumnsToScale])
    ScaledTestData[ColumnsToScale] = FittedScaler.transform(TestData[ColumnsToScale])
    return ScaledTrainData, ScaledTestData, FittedScaler

def GenerateNoisyTradingData(OriginalData, SmoothingMethod="EMA", SmoothingParams=None,
                             NoiseMethod="SimpleSampling", BlockSize=5, ApplyScaling=False):
    """
    Generate a noisy version of OHLCV trading data for a denoising autoencoder.
    
    Parameters:
        OriginalData (pd.DataFrame): Clean, chronologically ordered OHLCV data.
        SmoothingMethod (str): Smoothing method ("EMA", "SMA", or "SavGol").
        SmoothingParams (dict): Parameters for the chosen smoothing method.
        NoiseMethod (str): Noise generation method ("SimpleSampling" or "BlockSampling").
        BlockSize (int): Block size for block sampling (if applicable).
        ApplyScaling (bool): Whether to apply feature scaling after noise generation.
    
    Returns:
        tuple: (XNoisy, XOriginal, Scaler)
            XNoisy - Noisy data input to the autoencoder.
            XOriginal - The original target data.
            Scaler - The fitted scaler (if ApplyScaling is True) otherwise None.
    """
    # Step 1: Data Preparation
    # (Assumes OriginalData is clean and ordered; we work on a copy for processing)
    DataOriginal = OriginalData.copy()
    
    # Step 2: Smoothing the Data
    DataSmoothed = SmoothData(DataOriginal, SmoothingMethod, SmoothingParams)
    
    # Step 3: Calculate Residuals (High-Frequency Component)
    ResidualsData = CalculateResiduals(DataOriginal, DataSmoothed)
    
    # Step 4: Generate Noise from Residuals
    NoiseData = GenerateNoise(ResidualsData, NoiseMethod, BlockSize)
    
    # Step 5: Add Noise to Original Data
    NoisyDataIntermediate = AddNoise(DataOriginal, NoiseData)
    
    # Step 6: Enforce OHLCV Constraints
    DataNoisy = EnforceOHLCVConstraints(NoisyDataIntermediate)
    
    # Step 7: (Optional) Scaling
    FittedScaler = None
    if ApplyScaling:
        # We fit on the original clean data and transform both datasets
        DataNoisy, DataOriginal, FittedScaler = ScaleData(DataOriginal, DataNoisy)
    
    # Return final datasets for the autoencoder
    return DataNoisy, DataOriginal, FittedScaler