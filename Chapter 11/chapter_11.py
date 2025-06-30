# -*- coding: utf-8 -*-

# ----- page 5
import requests
from bs4 import BeautifulSoup

url = "https://biztoc.com/"
response = requests.get(url)

company_name = "NVIDIA"
ticker = "NVDA"

if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    news_items = soup.find_all('a', attrs={'data-p': True})

    for item in news_items:
        headline = item.get_text(strip=True)
        if company_name.lower() in headline.lower() or company_name.lower() in item['data-p'].lower() or ticker.lower() in item['data-p'].lower():
            url = item['href']
            print(f"Headline: {headline}")
            print(f"URL: {url}\n")
# ------
# ----- page 6
!pip install -q feedparser

import feedparser
from textblob import TextBlob

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity
# ------
# ----- page 6–7
stock_symbol = 'AAPL'
rss_url = f'https://finance.yahoo.com/rss/headline?s={stock_symbol}'

feed = feedparser.parse(rss_url)

data = []
for entry in feed.entries:
    data.append([entry.published_parsed, entry.title])

df = pd.DataFrame(data, columns=['date', 'title'])
df['date'] = df['date'].apply(lambda x: datetime(*x[:6]))

df['sentiment'] = df['title'].apply(get_sentiment)
# ------
# ----- page 8
print('minimum sentiment rating : ', np.min(df['sentiment']))
print(df.iloc[np.argmin(df['sentiment']) , :]['title'])

print('maximum sentiment rating : ', np.max(df['sentiment']))
df.iloc[np.argmax(df['sentiment']) , :]['title']
# ------
# ----- page 10
!pip install -q wikipedia-api

def get_sp500_company_links(wikipedia_url):
    response = requests.get(wikipedia_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})
    rows = table.findAll('tr')
    links = []
    for row in rows[1:]:
        link = row.find('td').find_next('td').find('a')['href']
        links.append(link)
    return links

wikipedia_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_company_links = get_sp500_company_links(wikipedia_url)
# ------
# ----- page 11
def get_wikipedia_text(page_title, language='en'):
    wiki = wikipediaapi.Wikipedia(language , headers = headers)
    page = wiki.page(page_title)
    if page.exists():
        return page.text
# ------
# ----- page 15
!pip install -qU sentence-transformers

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('average_word_embeddings_glove.840B.300d')

for title, content in page_content.items():
    embeddings[title] = model.encode(content, convert_to_tensor=True)

df_embeddings = pd.DataFrame(embeddings)
# ------
# ----- page 20
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim_matrix = cosine_similarity(df_embeddings.T)
df_cos_sim = pd.DataFrame(cosine_sim_matrix ,
    index = df_embeddings.columns,
    columns = df_embeddings.columns)
# ------
# ----- page 21
lbl = df_cos_sim.columns[2]
print('similarities for: ', lbl)
df_cos_sim.loc[lbl].sort_values(ascending=False)[1:5]

lbl = 'Goldman_Sachs'
pprint(df_cos_sim.loc[lbl].sort_values(ascending=False)[1:5])
# ------
# ----- page 22
import seaborn as sns
sns.clustermap(df_cos_sim)
# ------
# ----- page 24
import networkx as nx

distance_matrix = 1 - df_cos_sim
graph = nx.from_numpy_array(distance_matrix.values)

mst = nx.minimum_spanning_tree(graph)
# ------
# ----- page 27
min_indices = np.argmin(similarity_matrix, axis=1)

top_n_pair = 3
lst_dissim = select_and_rank_dissimilar_pairs(df_cos_sim.values ,
 df_cos_sim.columns , top_n_pair)

print(lst_dissim)
# ------
# ----- page 29
normalized_data = data[lst_dissim] / data[lst_dissim].iloc[0]
portfolio_dissim = np.sum(normalized_data * np.ones(len(lst_dissim)) * 1/len(lst_dissim), axis=1)
# ------
# ----- page 35
idx_train = int(80/100 * data.shape[0])
ret = data.iloc[:idx_train].pct_change().fillna(0)
print(f'Training end @ {data.iloc[idx_train].name}')

port = rp.Portfolio(returns = ret)
rf = 0
port.rf = rf
port.assets_stats(method_mu='hist', method_cov='hist')
port.rm = 'MV'
port.obj = 'Sharpe'

w_sharpe = port.optimization(model='Classic')
# ------
# ----- page 37
# HRP allocations are surrounded; Max Sharpe Ratio allocations are starred
# This is only a visualization reference for MST — no code given
# ------