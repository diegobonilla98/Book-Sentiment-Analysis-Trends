import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


file_path = "romeo_and_juliet.csv"
df = pd.read_csv(file_path)
df = df.drop(columns=["neutral"])
for col in df.columns:
    df[col] = savgol_filter(df[col], 51, 3)

sns.set_style("whitegrid")
emotions = df.columns[1:]
palette = sns.color_palette("husl", len(emotions))
fig, axs = plt.subplots(len(emotions), 1, figsize=(12, 4 * len(emotions)))
for ax, emotion, color in zip(axs, emotions, palette):
    ax.plot(df[emotion], color=color)
    ax.set_title(emotion, fontsize=16)
    ax.set_xlabel('Sentence Index', fontsize=14)
    ax.set_ylabel('Emotion Score', fontsize=14)
plt.tight_layout()
plt.savefig(file_path.replace(".csv", "_tight_plot.png"))


N = 4
variances = df.var().sort_values(ascending=False)
top_variances = variances[:N]
palette = sns.color_palette("Dark2", N)
plt.figure(figsize=(20, 10))
for emotion, color in zip(top_variances.index, palette):
    plt.plot(df[emotion], color=color, alpha=0.6, label=emotion, linewidth=2.5)
plt.title('Romeo and Juliet. Top {} Emotions Trend'.format(N), fontsize=18)
plt.xlabel('Sentence Index', fontsize=14)
plt.ylabel('Emotion Score', fontsize=14)
plt.legend(loc='upper left', title="Emotions")
plt.savefig(file_path.replace(".csv", "_join_plot.png"))
