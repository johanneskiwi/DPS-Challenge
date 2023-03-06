import pandas as pd


def load_dataset(file):
    # Load data into pandas dataframe
    data = pd.read_csv(file)

    # Only use first 5 Categories:
    df = data.iloc[:, : 5]

    return df


def preprocess(df, horizon=2021, for_visuals=False):
    # Drop all rows with entry "Summe"
    df = df.drop(df[df.MONAT == "Summe"].index)

    # Drop all rows that are not from type "insgesamt"
    df = df.drop(df[df.AUSPRÄGUNG != "insgesamt"].index)

    # Drop all rows with year > 2021
    df = df.drop(df[df.JAHR > horizon].index)

    # Rename columns
    df = df.rename(columns={'MONAT': 'date'})
    df = df.rename(columns={'WERT': 'value'})

    df['date'] = pd.to_datetime(df['date'], format='%Y%m')

    # Drop columns
    if for_visuals:
        df = df.drop(["AUSPRÄGUNG", "JAHR"], axis=1)
        df = df.rename(columns={'MONATSZAHL': 'category'})

    else:
        # Drop all rows that do not belong to category "Alkoholunfälle"
        df = df.drop(df[df.MONATSZAHL != "Alkoholunfälle"].index)

        df = df.drop(["MONATSZAHL", "AUSPRÄGUNG", "JAHR"], axis=1)

        # Change type of "date" column to datetime
        df['date'] = pd.to_datetime(df['date'], format='%Y%m')

        df = sort_data(df)

        # Check for missing values
        print(f'Number of rows with missing values: {df.isnull().any(axis=1).mean()}')

    return df


def sort_data(df):
    df = df.set_index('date', drop=False)
    df = df.asfreq('MS')
    df = df.sort_index()
    return df
