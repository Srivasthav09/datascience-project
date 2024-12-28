import pandas as pd
import numpy as np
data = pd.read_csv("flipkart_dummy.csv")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from transformers import pipeline


def _give_sentiment(sentence):

  text = []

  aspect_list = ['phone',
 'camera',
 'battery',
 'price',
 'product',
 'performance',
 'quality',
 'display']

  for aspect in aspect_list:
    if aspect in sentence:
      inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
      outputs = absa_model(**inputs)
      probs = F.softmax(outputs.logits, dim=1)
      probs = probs.detach().numpy()[0]
      sentiment_index = np.argmax(probs)

      if sentiment_index == 0:
        text.append(sentence+"aaaa"+aspect+"xxxx0")
      elif sentiment_index == 2:
        text.append(sentence+"aaaa"+aspect+"xxxx1")

      else:
        pass

  return str(text)


# Load Aspect-Based Sentiment Analysis model
absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1", use_fast=False)
absa_model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")

# Load a traditional Sentiment Analysis model
sentiment_model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_path, tokenizer=sentiment_model_path)

data = data[['processed_review']]

data['text_aspect_sentiment'] = data['processed_review'].apply(_give_sentiment)

data.to_csv('data.csv', index=False)