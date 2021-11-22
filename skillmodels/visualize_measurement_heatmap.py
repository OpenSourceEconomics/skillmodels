"""Visualize the correlation heatmap of measurement variables"""
import pandas as pd
import numpy as np 
import mathplotlib.pyplot as plt
import seaborn as sns
from skillmodels.process_model import process_model

def plot_period_heatmap(period, model_dict, data):
    model = process_model(model_dict)

    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(6)
    fig.set_figwidth(11)
    df = df.xs(age, level="age_bin")
    corr = df[measurements].corr().round(2)
    labels = corr.replace(1, "")
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        annot=labels,
        fmt="",
        ax=ax,
        annot_kws={"fontsize": 11},
    )
    ax.set_title(f"Age: {age}")
    return ax