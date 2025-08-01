import pandas as pd
import polars as pl
import numpy as np
import pickle, warnings, datetime
warnings.filterwarnings('ignore')

df = pl.read_csv('./data/processed/gdelt_combined_20250610_1716.csv')

df.shape

v_cols_drop = [i for i in df.columns if i.startswith("v") and "v19" in i]

# http://data.gdeltproject.org/documentation/GCAM-MASTER-CODEBOOK.TXT
c_columns = [i for i in df.columns if i.startswith("c")]
c_cols_to_keep = [
    "c3.","c4.1;","c4.16","c6.","c16.60","c41.","c18.1;","c18.2;","c18.21;","c18.30;","c18.33;","c18.50;","c18.66;","c18.68;","c18.83;","c18.121;","c18.137;","c18.157;","c18.164;","c18.254;",
]
c_cols_keep = []
for c in c_cols_to_keep:
    c_cols_keep.extend([i for i in c_columns if c in i])
c_cols_keep.extend([i for i in c_columns if "c18." in i and "ECON_" in i])
c_cols_drop = [i for i in c_columns if i not in c_cols_keep]

df = df.drop(v_cols_drop)
df = df.drop(c_cols_drop)

# For each column, drop if all values are equal to 0
for col in df.columns:
    if df[col].min() == df[col].max():
        df = df.drop(col)
df.shrink_to_fit(in_place=True)

# Convert to smaller data types
df = df.select(pl.all().shrink_dtype())

# Remove a list of titles that are probably home pages or other non-article content
titles_to_remove = [
    'News briefs',
    'Latest Articles',
    'The Nashville Ledger',
    'Today in History',
    'National News - Media One Radio Group (WWSE | WJTN | WHUG | WKSN | WQFX',
    "Aero-News Network: The aviation and aerospace world's daily/real-time news and information service",
    'Drake & 21 Savage Add More Texas Concert Dates Due To High Demand',
    'Stock Market | FinancialContent Business Page',
    'Radio Station WHMI 93.5 FM &#x2014; Livingston County Michigan News, Weather, Traffic, Sports, School Updates, and the Best Classic Hit',
    'National - KSYL-AM',
    'Business Highlights',
    'Business Highlights',
    'National News - 1540 WADK Newport',
    'National - Carroll Broadcasting Inc.',
    'ABC National - WOND',
    "Breaking National News - 92.7-FM TheDRIVE - Bob & Tom Mornings, Central New York's Best Rock All Day",
    "ABC - National News - Xtra 99.1 FM - Today's Hits and Yesterday's Favorites",
    "Ed Bruce, Legendary Country Songwriter, 'Maverick' Actor, Dead At 81",
    "ABC National News - Beach 95.1 - WBPC Panama City Beach Greatest Hits of the 60s, 70s & 80s",
    "CES gadget show: How watching TV will change in the 2020s",
    "Despite business warnings, GOP moves ahead with voting bills",
    "KTBB.com - News Weather Talk",
    "AP Story",
    "SRN - US News - Taylorville Daily News",
]

# Replace empty article titles with z
df = df.with_columns(
    pl.col('article_title').fill_null('z')
)

df = df.filter(
    ~df['article_title'].is_in(titles_to_remove)
)

# Drop where article_title contains 'AP News in Brief'
df = df.filter(
    ~df['article_title'].str.contains('AP News in Brief at', literal=True)
    )

# Filter to reliable news sites. I don't want to limit to a certain set of sites because I want to keep local news in

# The idea is to remove sites that are unlikely to report about price-influencing stories. For example, some sites report about cheap flights or travel points
sites_to_keep = [
'yahoo.com','msn.com','fool.com','reuters.com','seekingalpha.com','themarketsdaily.com','forbes.com','investing.com','cnn.com','marketscreener.com','washingtonpost.com','nytimes.com','investors.com','tickerreport.com','insidermonkey.com','morningstar.com','abc7news.com','businessinsider.com','prnewswire.com','bnnbloomberg.ca','zerohedge.com','nasdaq.com','marketwatch.com','abc7ny.com','streetinsider.com','apnews.com','econintersect.com','foxbusiness.com','cnbc.com'
]

df = df.filter(
    df['V2SOURCECOMMONNAME'].is_in(sites_to_keep)
)

# %%
df=df.to_pandas()
df

# Extracting headlines from URLs when article title is empty
df['V2DOCUMENTIDENTIFIER'].value_counts()

df['url'] = df['V2DOCUMENTIDENTIFIER']
# Remove the protocol (http:// or https://) and the domain name
df['url'].replace(r'^(https?://)', '', regex=True, inplace=True)

# remove anything before .com, .org, .net, etc.
df['url'].replace(r'^[^/]+/', '', regex=True, inplace=True)

# remove strings of atleast 7 numbers
df['url'].replace(r'\d{7,}', '', regex=True, inplace=True)


words_to_remove = ['news/', 'article/', 'forum/', 'entertainment/', 'stories/', 'national/',
                   'national_news/', 'story/', 'travel/', 'articles/', 'us/', 'world/',
                   'world-news', 'blog/', 'nation-world/', 'region/', 'post/', 'recommends/',
                   'headlines/', 'business/', 'ap/', 'business-economy/', '.html', '.htm', 'x/'
]

for word in words_to_remove:
    df['url'].replace(word, '', regex=True, inplace=True)

# remove anything that looks like a date
df['url'].replace(r'\d{4}/\d{2}/\d{2}', '', regex=True, inplace=True)
df['url'].replace(r'\d{4}-\d{2}-\d{2}', '', regex=True, inplace=True)
df['url'].replace(r'20[2][0-9][01][0-9][0-9]{2}', '', regex=True, inplace=True)

df['url'].replace(r'/', ' ', regex=True, inplace=True)
df['url'].replace(r'-', ' ', regex=True, inplace=True)
df['url'].replace(r'_', ' ', regex=True, inplace=True)
df['url'].replace(r'\.', ' ', regex=True, inplace=True)
df['url'].replace(r'\?', ' ', regex=True, inplace=True)
df['url'].replace(r'  ', ' ', regex=True, inplace=True)

df['url']=df['url'].str.lstrip()
df['url']=df['url'].str.rstrip()
df['url']=df['url'].str.lower()

words_to_remove = ['syndicated id=', 'article', 'usubmit', 'nation article', 'ap', 'nation ', 
                   'news briefs t=','content','viewtopic php f=3&t=',' cfm c_id=3&objectid=',' cfm c_id=2&objectid=',
                   'national','latest','cfm c_id=7&objectid=','story aspx id=','post_type=news&p=','latest','world us canada',
                   'npr story storyid=','p=','tag * index more=','latest article'
]

for word in words_to_remove:
    df['url'] = np.where(df['url']==word, '', df['url'])

df['url'].replace(r'zz ', '', regex=True, inplace=True)

df['url']=df['url'].str.lstrip()
df['url']=df['url'].str.rstrip()

# replace article title with url if article title is empty
df['article_title'] = df['article_title'].str.lower()
df['article_title'] = np.where(df['article_title'] == 'z', df['url'], df['article_title'])

df['article_title'].replace(r'/', ' ', regex=True, inplace=True)
df['article_title'].replace(r'-', ' ', regex=True, inplace=True)
df['article_title'].replace(r'\.', ' ', regex=True, inplace=True)
df['article_title'].replace(r'\?', ' ', regex=True, inplace=True)
df['article_title'].replace(r"'", ' ', regex=True, inplace=True)
df['article_title'].replace(r",", ' ', regex=True, inplace=True)

# remove strings of at least 8 characters that contain both letters and numbers
df['article_title'].replace(r'\b(?=\w*[a-zA-Z])(?=\w*[0-9])\w{8,}\b', ' ', regex=True, inplace=True)

df['article_title'].replace(r' +', ' ', regex=True, inplace=True)

df['article_title'] = np.where(df['article_title'] == 'z', '', df['article_title'])
df['article_title'] = df['article_title'].str.strip()
df['article_title'] = df['article_title'].str.lower()

df.drop(columns=['url'], inplace=True)

# Drop records where article title starts with
for s in ['article cfm c id=','external php s=','starttime=','post type=news&p=',
          'h article=','p=','page=','default aspx','syndicated id=','article aspx id=']:
    df = df[~df['article_title'].str.startswith(s, na=False)]

for word in ['national article','business','local article','national',
             'news and closings national','abc business','abc','national hits fm',
             'world hits fm','story','id']:
    df = df[df['article_title'] != word]

# Drop records where article title is all numbers
df = df[~df['article_title'].str.match(r'^\s*[0-9]+(\s+[0-9]+)*\s*$', na=False)]

# Drop records where article title is empty
df = df[~df['article_title'].str.strip().eq('')]

# Random popular article that mentions an airline but as an aside
# https://nationalpost.com/pmn/news-pmn/mighty-mississippi-scientists-use-model-in-land-loss-fight
df = df[~df['article_title'].str.contains('mighty mississippi')]
df = df[~df['article_title'].str.contains('mississippi model')]

df = df[~df['article_title'].str.contains('ted kaczynski')]
df = df[~df['article_title'].str.contains('sexually harassed')]

# Remove 9/11 and related historical articles
for word in ['9 11','september 11','sept 11','on this day','the year in','year in review','lessons learned in','top stories']:
    df = df[~df['article_title'].str.contains(word, na=False)]

for word in ['history','historical','today in history']:
    df = df[~df['article_title'].str.startswith(word, na=False)]

df.drop_duplicates(subset=['GKGRECORDID'], inplace=True)
df.shape

list(df.columns)

# Save df to pickle file
with open('./data/processed/gdelt_intermediate_cleaned_finance.pkl', 'wb') as f:
    pickle.dump(df, f)
