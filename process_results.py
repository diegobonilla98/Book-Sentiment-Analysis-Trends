import pandas as pd
import pickle


file_path = "romeo_and_juliet.pkl"

with open(file_path, 'rb') as handle:
    data = pickle.load(handle)

emotions = ['gratitude', 'admiration', 'curiosity', 'neutral', 'approval', 'confusion', 'annoyance', 'excitement', 'caring', 'relief', 'disappointment', 'joy', 'disgust', 'amusement', 'anger', 'disapproval', 'desire', 'surprise', 'remorse', 'love', 'optimism', 'grief', 'sadness', 'realization', 'embarrassment', 'fear', 'pride', 'nervousness']
df = pd.DataFrame(columns=emotions)
for row in data:
    emotions_dict = dict(zip(emotions, [None for _ in range(len(emotions))]))
    for emotion in row:
        emotions_dict[emotion["label"]] = emotion["score"]
    df = df.append(emotions_dict, ignore_index=True)

df.to_csv(file_path.replace(".pkl", ".csv"), index=False)
