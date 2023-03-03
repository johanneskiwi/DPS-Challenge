from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

PLOT = True

# Load data into pandas dataframe
data = pd.read_csv(r'data.csv')

# Only use first 5 Categories:
df = data.iloc[:, : 5]

# Skip recent values (Year > 2020)
df = df[df["JAHR"] <= 2020]

# For visualization get only one value ("Summe") for each year
df_compr = df[df["MONAT"] == "Summe"]

# Use only subcategory "insgesamt"
df_compr = df_compr[df_compr["AUSPRÄGUNG"] == "insgesamt"]

# Drop columns "AUSPRÄGUNG" and "MONAT"
df_compr = df_compr.drop(["AUSPRÄGUNG", "MONAT"], axis=1)

if PLOT:
    sns.barplot(x="JAHR",  # x variable name
                y="WERT",  # y variable name
                hue="MONATSZAHL",  # group variable name
                data=df_compr  # dataframe to plot
                )
    plt.show()


