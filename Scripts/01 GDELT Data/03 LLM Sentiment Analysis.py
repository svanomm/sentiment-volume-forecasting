import pickle, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

with open('./data/processed/gdelt_intermediate_cleaned_finance.pkl', 'rb') as f:
    df = pickle.load(f)
df = df[['GKGRECORDID','article_title']]

# Load model: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", model_kwargs={"torch_dtype": "float16"})
prompt = "Instruct: Extract the sentiment from this news headline that is most likely to affect the airline stock market\nQuery:"

# Encode the article titles with the model
query_embeddings = model.encode(
    df['article_title'], 
    prompt=prompt,
    show_progress_bar=True,
    batch_size=64,
    truncate_dim=32
)

query_embeddings = np.array(query_embeddings)

for i in range(query_embeddings.shape[1]):
    df[f'llm_dimension_{i:02d}'] = query_embeddings[:, i]

# Save df_sentiment to a pickle file
df.drop(columns=['article_title'], inplace=True)

with open('./data/processed/gdelt_llm_sentiment_finance.pkl', 'wb') as f:
    pickle.dump(df, f)
