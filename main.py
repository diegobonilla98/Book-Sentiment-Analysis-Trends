import tqdm
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
import re
import pickle


tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")

emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

file_path = "dracula.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

all_emotions = []

chunks = text.split(".")
for chunk in tqdm.tqdm(chunks, smoothing=0.05):
    clean_chunk = chunk.replace('\n', ' ')
    clean_chunk = re.sub(' +', ' ', clean_chunk)
    clean_chunk = clean_chunk.strip()
    if len(clean_chunk) < 25:
        continue
    emotion_labels = emotion(clean_chunk, top_k=28)
    all_emotions.append(emotion_labels)

with open(file_path.replace(".txt", ".pkl"), 'wb') as handle:
    pickle.dump(all_emotions, handle, protocol=pickle.HIGHEST_PROTOCOL)

