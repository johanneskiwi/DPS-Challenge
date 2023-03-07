from matplotlib import pyplot as plt
import pandas as pd

from utils import load_dataset, preprocess, sort_data

LINE_PLOT = True


def plot_input_data(file):
    """Creates plots for input data"""

    labels = ["Alkoholunfälle", "Fluchtunfälle", "Verkehrsunfälle"]
    colors = ["r", "g", "y"]

    # Load data into pandas dataframe
    df = load_dataset(file)

    df = preprocess(df, horizon=2020, for_visuals=True)

    df_alk = sort_data(df[df["category"] == labels[0]])
    df_flucht = sort_data(df[df["category"] == labels[1]])
    df_verkehr = sort_data(df[df["category"] == labels[2]])

    fig1, axs = plt.subplots(3)
    axs[0].plot(df_alk["ds"], df_alk["y"], color=colors[0])
    axs[1].plot(df_flucht["ds"], df_flucht["y"], color=colors[1])
    axs[2].plot(df_verkehr["ds"], df_verkehr["y"], color=colors[2])

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
        plt.plot(l["ds"], l["y"], color=c)

    plt.xlim(pd.Timestamp('2000-01-01'), pd.Timestamp('2020-12-01'))
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel("Number of Accidents")
    plt.title("Number of accidents per category", fontsize=14)
    plt.legend(lines, labels=labels, loc="center right")

    plt.show()


def plot_predictions(train_data, test_data, pred_data, steps=24):
    """Plot prediction horizon compared with test data"""

    fig, ax = plt.subplots(figsize=(9, 4))
    train_data['y'].plot(ax=ax, label='train')
    test_data['y'][:steps].plot(ax=ax, label='test')
    pred_data.plot(ax=ax, label='predictions')

    plt.xlabel("Date")
    plt.ylabel("Number of Accidents")
    plt.title("Training, Test and Prediction Data")

    ax.legend()
    plt.show()
