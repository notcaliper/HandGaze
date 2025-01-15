import json
import os
from typing import List, Dict, Optional
from spellchecker import SpellChecker

class OfflineDictionary:
    def __init__(self):
        self.dictionary_file = 'dictionary_data/english_words.json'
        self.word_dict = self.load_dictionary()
        self.spell = SpellChecker()
        
    def load_dictionary(self) -> Dict:
        """Load the dictionary from file or create if not exists"""
        os.makedirs('dictionary_data', exist_ok=True)
        
        if os.path.exists(self.dictionary_file):
            with open(self.dictionary_file, 'r') as f:
                return json.load(f)
        else:
            # Create basic dictionary from SpellChecker's word frequency
            word_dict = {}
            for word in self.spell.word_frequency.dictionary:
                word_dict[word] = {
                    'word': word,
                    'frequency': self.spell.word_frequency.dictionary[word]
                }
            
            # Save dictionary
            with open(self.dictionary_file, 'w') as f:
                json.dump(word_dict, f)
            
            return word_dict
    
    def get_word_info(self, word: str) -> Optional[Dict]:
        """Get information about a word"""
        return self.word_dict.get(word.lower())
    
    def get_suggestions(self, word: str) -> List[str]:
        """Get spelling suggestions for a word"""
        suggestions = list(self.spell.candidates(word))
        # Sort suggestions by frequency
        return sorted(suggestions, 
                     key=lambda x: self.word_dict.get(x, {}).get('frequency', 0),
                     reverse=True)[:3]
    
    def is_valid_word(self, word: str) -> bool:
        """Check if a word is valid"""
        return word.lower() in self.word_dict
    
    def add_word(self, word: str, info: Dict = None):
        """Add a new word to the dictionary"""
        if info is None:
            info = {'word': word}
        self.word_dict[word.lower()] = info
        self._save_dictionary()
    
    def _save_dictionary(self):
        """Save the dictionary to file"""
        with open(self.dictionary_file, 'w') as f:
            json.dump(self.word_dict, f)

# Create dictionary data file if running this script directly
if __name__ == "__main__":
    dictionary = OfflineDictionary()
    print(f"Dictionary loaded with {len(dictionary.word_dict)} words")
    
    # Test some functionality
    test_word = "hello"
    print(f"\nTesting word: {test_word}")
    print(f"Is valid: {dictionary.is_valid_word(test_word)}")
    print(f"Info: {dictionary.get_word_info(test_word)}")
    
    misspelled = "helllo"
    print(f"\nTesting misspelled word: {misspelled}")
    print(f"Suggestions: {dictionary.get_suggestions(misspelled)}")
