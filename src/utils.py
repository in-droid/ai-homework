from typing import Tuple
import re 

import requests
import pandas as pd
from sklearn.model_selection import train_test_split



def split_data(df: pd.DataFrame, test_size:float=0.3) ->Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,                  
        stratify=df['CATEGORY'],
        random_state=42
    )

    return train_df, test_df


def get_categories(df: pd.DataFrame) -> list:
    categories = df['CATEGORY'].unique().tolist()
    return categories

def wiki_intro(name: str) -> str:
    def clean_name(name):
        name = re.sub(r'&', 'and', name)
        name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
        return name.strip()

    def fetch_intro(title):
        S = requests.Session()
        PARAMS = {
            'action': 'query',
            'prop': 'extracts',
            'exintro': '',
            'explaintext': '',
            'titles': title,
            'format': 'json',
            'redirects': 1
        }
        R = S.get('https://en.wikipedia.org/w/api.php', params=PARAMS).json()
        page = next(iter(R['query']['pages'].values()))
        return page.get('extract', '')

    # Try direct title lookup
    cleaned_name = clean_name(name)
    content = fetch_intro(cleaned_name)

    # Fallback: search API if nothing was returned
    if not content:
        print(f"[Fallback] Searching for: {name}")
        search_params = {
            'action': 'query',
            'list': 'search',
            'srsearch': name,
            'format': 'json'
        }
        search_result = requests.get('https://en.wikipedia.org/w/api.php', 
                                     params=search_params).json()
        search_hits = search_result.get('query', {}).get('search', [])
        if search_hits:
            top_title = search_hits[0]['title']
            content = fetch_intro(top_title)

    # If nothing was found, return an empty string
    if not content:
        print(f"No Wikipedia content found for: {name}")
        return ''

    return content.strip()
