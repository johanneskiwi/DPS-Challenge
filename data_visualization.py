from matplotlib import pyplot as plt
import pandas as pd

from utils import load_dataset, preprocess, sort_data

LINE_PLOT = True


def plot_data(file):
    labels = ["Alkoholunfälle", "Fluchtunfälle", "Verkehrsunfälle"]
    colors = ["r", "g", "y"]

    # Load data into pandas dataframe
    df = load_dataset(file)

    df = preprocess(df, horizon=2020, for_visuals=True)

    df_alk = sort_data(df[df["category"] == labels[0]])
    df_flucht = sort_data(df[df["category"] == labels[1]])
    df_verkehr = sort_data(df[df["category"] == labels[2]])

    fig1, axs = plt.subplots(3)
    axs[0].plot(df_alk["date"], df_alk["value"], color=colors[0])
    axs[1].plot(df_flucht["date"], df_flucht["value"], color=colors[1])
    axs[2].plot(df_verkehr["date"], df_verkehr["value"], color=colors[2])

    for ax in axs:
        ax.set_xlim(pd.Timestamp('2000-01-01'), pd.Timestamp('2020-12-01'))
        ax.grid()
        ax.set_ylabel("Number of Accidents")

    lines = [df_alk, df_flucht, df_verkehr]
    fig1.legend(lines, labels=labels, loc="lower right")

    axs[2].set_xlabel("Date")
    fig1.suptitle("Number of accidents per category", fontsize=14)

    plt.figure()
    for l, c in zip(lines, colors):
        plt.plot(l["date"], l["value"], color=c)

    plt.xlim(pd.Timestamp('2000-01-01'), pd.Timestamp('2020-12-01'))
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel("Number of Accidents")
    plt.title("Number of accidents per category", fontsize=14)
    plt.legend(lines, labels=labels, loc="center right")

    plt.show()


if __name__ == "__main__":
    plot_data(r'data.csv')