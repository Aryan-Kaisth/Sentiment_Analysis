import re
import emoji
from nltk.stem import WordNetLemmatizer
import unicodedata

def remove_urls(text):
                return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

# Convert emojis to text descriptions
def convert_emojis_to_text(text):
    return emoji.demojize(text)

# Remove punctuation and numbers
def remove_punct_numbers(text):
    return re.sub(r'[^a-z\s]', '', text)

def lemmatize_words(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

def remove_accents_diacritics(text):
    """
    Normalize unicode text and remove accents/diacritics.
    Example: 'café' → 'cafe'
    """
    # Normalize to 'NFKD' form (decomposes accents)
    text = unicodedata.normalize('NFKD', text)
    # Keep only base characters (ignore diacritics)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    return text