import re
import nltk
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

stop_words = nltk.corpus.stopwords.words('english')


def clean_text(text:str)->str:
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text= text.lower()
    text = " ".join([word for word in nltk.word_tokenize(text) if word not in  stop_words])
    return text

def apply_text_cleaning(df:pd.DataFrame,text_column:str='text')->pd.DataFrame:
    df['clean_text'] = df[text_column].apply(clean_text)
    return df


def normalize_metadata(df: pd.DataFrame, metadata_columns: list) -> pd.DataFrame:
    scalar = MinMaxScaler()
    df[metadata_columns] = scalar.fit_transform(df[metadata_columns])
    return df

def extract_url_metadata(df: pd.DataFrame, url_column: str = 'news_url') -> pd.DataFrame:
    """Extract metadata features from the URL column."""
    # Ensure column exists and handle NaNs
    df[url_column] = df[url_column].fillna('')
    
    # URL Length
    df['url_length'] = df[url_column].apply(len)
    
    # HTTPS Presence
    df['is_https'] = df[url_column].apply(lambda x: 1 if 'https' in str(x) else 0)
    
    # Count of special characters (simple proxy for complexity)
    df['url_special_chars'] = df[url_column].apply(lambda x: len(re.findall(r'[^A-Za-z0-9\s]', str(x))))
    
    # Domain extraction (simplified for feature count)
    # We could do more complex domain analysis, but for now let's just use length of domain
    # or just rely on the above features. 
    # Let's add count of digits as well.
    df['url_digits'] = df[url_column].apply(lambda x: len(re.findall(r'[0-9]', str(x))))

    return df