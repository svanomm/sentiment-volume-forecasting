import pickle
import pandas as pd

# Read "./data/GCAM-MASTER-CODEBOOK.TXT" in latin-1 encoding
codebook = pd.read_csv("./data/GCAM-MASTER-CODEBOOK.TXT", sep="\t", encoding='latin-1')

with open('./Data/Processed/GDELT_Clean_finance.pkl', 'rb') as f:
    df = pickle.load(f)

cols = [i for i in df.columns if (i.startswith("c") or i.startswith("v")) and 'cum' not in i]
cols = [i.replace('_lag01', '') for i in cols]
cols = [i.split(';')[0] for i in cols]

codebook = codebook[codebook['Variable'].isin(cols)]
codebook = codebook[['Variable', 'Type', 'DictionaryHumanName', 'DimensionHumanName', 'DictionaryCitation']]
codebook['desc'] = codebook['Variable'] + ', ' + codebook['Type'] + ', ' + codebook['DictionaryHumanName'] + ', ' + codebook['DimensionHumanName']

with open('./output/gdelt_metrics_used.txt', 'w') as f:
    for i in sorted(codebook['desc'].tolist()):
        f.write(i + '\n')
        
with open('./output/gdelt_metrics_sources.txt', 'w') as f:
    for i in sorted(codebook['DictionaryCitation'].unique().tolist()):
        f.write(i + '\n')