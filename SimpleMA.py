import pandas as pd
import pandas_ta as ta

import pandas as pd
import pandas_ta as ta

def GenerateSmaCrossoverSignal(DataFrame: pd.DataFrame, ShortWindow: int = 10, LongWindow: int = 30) -> pd.DataFrame:
    """
    Add SMA crossover signals with price confirmation to the OHLCV DataFrame.

    Parameters:
    - DataFrame: A pandas DataFrame with a 'Close' column.
    - ShortWindow: Period for short SMA.
    - LongWindow: Period for long SMA.

    Returns:
    - DataFrame with 'SmaShort', 'SmaLong', and 'Signal' columns.
    """
    
    # Calculate short and long SMAs
    DataFrame["SmaShort"] = ta.sma(DataFrame["Close"], length=ShortWindow)
    DataFrame["SmaLong"] = ta.sma(DataFrame["Close"], length=LongWindow)

    # Initialize signal column
    DataFrame["Signal"] = 0

    # Buy condition: crossover up + close > short SMA
    BuySignal = (
        (DataFrame["SmaShort"] > DataFrame["SmaLong"]) &
        (DataFrame["SmaShort"].shift(1) <= DataFrame["SmaLong"].shift(1)) &
        (DataFrame["Close"] > DataFrame["SmaShort"])
    )

    # Sell condition: crossover down + close < short SMA
    SellSignal = (
        (DataFrame["SmaShort"] < DataFrame["SmaLong"]) &
        (DataFrame["SmaShort"].shift(1) >= DataFrame["SmaLong"].shift(1)) &
        (DataFrame["Close"] < DataFrame["SmaShort"])
    )

    # Assign signals
    DataFrame.loc[BuySignal, "Signal"] = 1
    DataFrame.loc[SellSignal, "Signal"] = -1

    return DataFrame
