import json
import os
import requests
from typing import Dict
from tqdm import tqdm

def download_word_list():
    """Download a comprehensive word list"""
    # Using word list from github
    url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_dictionary.json"
    response = requests.get(url)
    return json.loads(response.text)

def create_enhanced_dictionary() -> Dict:
    """Create an enhanced dictionary with additional word information"""
    print("Downloading word list...")
    words = download_word_list()
    
    enhanced_dict = {}
    print("\nProcessing words...")
    for word in tqdm(words.keys()):
        # Basic word information
        enhanced_dict[word.lower()] = {
            'word': word,
            'length': len(word),
            'frequency': 1  # Basic frequency, can be improved
        }
    
    # Save the enhanced dictionary
    os.makedirs('dictionary_data', exist_ok=True)
    with open('dictionary_data/english_words.json', 'w') as f:
        json.dump(enhanced_dict, f)
    
    print(f"\nDictionary created with {len(enhanced_dict)} words")
    return enhanced_dict

if __name__ == "__main__":
    create_enhanced_dictionary()
