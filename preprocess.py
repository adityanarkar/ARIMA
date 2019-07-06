import pandas as pd


def getBuySellCalls(df: pd.DataFrame, colName: str):
    df["temp"] = df[colName].shift(1)
    print(df.head())
    df["Expected"] = df["temp"] >= df[colName]
    df["Expected"] = df["Expected"].map(lambda x: isTrue(x))
    print(df.tail())
    # df.dropna(inplace=True)
    return df


def isTrue(x: bool):
    if x:
        return 1
    return 0
