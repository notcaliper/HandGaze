import json
import os
import requests
from typing import List, Dict, Optional, Set
from spellchecker import SpellChecker
from functools import lru_cache
from collections import defaultdict
from tqdm import tqdm

class OfflineDictionary:
    def __init__(self, create_new: bool = False):
        self.dictionary_file = 'dictionary_data/english_words.json'
        self.word_dict = {}
        self.word_set = set()  # For faster lookups
        self.spell = SpellChecker()
        self.pending_updates = False
        
        if create_new:
            self.create_enhanced_dictionary()
        else:
            self.load_dictionary()
        
    def load_dictionary(self) -> None:
        """Load the dictionary from file or create if not exists"""
        os.makedirs('dictionary_data', exist_ok=True)
        
        if os.path.exists(self.dictionary_file):
            # Memory efficient loading using generator
            with open(self.dictionary_file, 'r') as f:
                data = json.load(f)
                # Process in chunks for memory efficiency
                self.word_dict = data
                self.word_set = set(data.keys())
        else:
            # Create optimized dictionary from SpellChecker's word frequency
            frequency_dict = self.spell.word_frequency.dictionary
            self.word_dict = {
                word: {
                    'word': word,
                    'frequency': freq
                }
                for word, freq in frequency_dict.items()
            }
            self.word_set = set(self.word_dict.keys())
            self._save_dictionary()
    
    def download_word_list(self) -> Dict:
        """Download a comprehensive word list"""
        url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_dictionary.json"
        response = requests.get(url)
        return json.loads(response.text)
    
    def create_enhanced_dictionary(self) -> None:
        """Create an enhanced dictionary with additional word information"""
        print("Downloading word list...")
        words = self.download_word_list()
        
        print("\nProcessing words...")
        for word in tqdm(words.keys()):
            # Enhanced word information
            word_lower = word.lower()
            self.word_dict[word_lower] = {
                'word': word,
                'length': len(word),
                'frequency': self.spell.word_frequency.dictionary.get(word_lower, 1)
            }
        
        self.word_set = set(self.word_dict.keys())
        self._save_dictionary()
        print(f"\nDictionary created with {len(self.word_dict)} words")
    
    @lru_cache(maxsize=1000)
    def get_word_info(self, word: str) -> Optional[Dict]:
        """Get information about a word with caching"""
        return self.word_dict.get(word.lower())
    
    @lru_cache(maxsize=1000)
    def get_suggestions(self, word: str) -> List[str]:
        """Get spelling suggestions for a word with caching"""
        if self.is_valid_word(word):
            return [word]
            
        candidates = list(self.spell.candidates(word))
        # Use defaultdict for faster frequency lookups
        freq_dict = defaultdict(int)
        for candidate in candidates:
            freq_dict[candidate] = self.word_dict.get(candidate, {}).get('frequency', 0)
        
        # Sort only the top N candidates for efficiency
        TOP_N = 3
        return sorted(candidates, 
                     key=lambda x: freq_dict[x],
                     reverse=True)[:TOP_N]
    
    def is_valid_word(self, word: str) -> bool:
        """Check if a word is valid using set for O(1) lookup"""
        return word.lower() in self.word_set
    
    def add_word(self, word: str, info: Dict = None):
        """Add a new word to the dictionary"""
        word = word.lower()
        if info is None:
            info = {'word': word}
        self.word_dict[word] = info
        self.word_set.add(word)
        self.pending_updates = True
        
        # Clear relevant caches
        self.get_word_info.cache_clear()
        self.get_suggestions.cache_clear()
        
        # Batch save: only save if there are no pending operations
        if len(self.word_dict) % 10 == 0:
            self._save_dictionary()
    
    def _save_dictionary(self):
        """Save the dictionary to file if there are pending updates"""
        if not self.pending_updates:
            return
            
        with open(self.dictionary_file, 'w') as f:
            json.dump(self.word_dict, f)
        self.pending_updates = False
    
    def __del__(self):
        """Ensure pending updates are saved when object is destroyed"""
        if self.pending_updates:
            self._save_dictionary()

# Create dictionary data file if running this script directly
if __name__ == "__main__":
    dictionary = OfflineDictionary(create_new=True)
    print(f"Dictionary loaded with {len(dictionary.word_dict)} words")
    
    # Test some functionality
    test_word = "hello"
    print(f"\nTesting word: {test_word}")
    print(f"Is valid: {dictionary.is_valid_word(test_word)}")
    print(f"Info: {dictionary.get_word_info(test_word)}")
    
    misspelled = "helllo"
    print(f"\nTesting misspelled word: {misspelled}")
    print(f"Suggestions: {dictionary.get_suggestions(misspelled)}")
