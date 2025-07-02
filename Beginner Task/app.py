import re
import string
from collections import defaultdict, Counter
from flask import Flask, render_template, request, jsonify

import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize

# Download required NLTK data only if not already available
def download_nltk_data():
    try:
        nltk.data.find('corpora/brown')
    except LookupError:
        nltk.download('brown')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

download_nltk_data()

# Main Autocorrect Class
class AutocorrectKeyboard:
    def __init__(self, n=3):
        self.n = n
        self.ngram_model = defaultdict(Counter)
        self.vocab = set()
        self.word_counts = Counter()
        self.build_model()

    def build_model(self):
        """Build n-gram model using Brown corpus"""
        sentences = brown.sents()
        for sentence in sentences:
            words = [word.lower() for word in sentence if word.isalpha()]
            self.vocab.update(words)
            self.word_counts.update(words)
            for i in range(len(words) - self.n + 1):
                ngram = tuple(words[i:i + self.n - 1])
                next_word = words[i + self.n - 1]
                self.ngram_model[ngram][next_word] += 1

    def edit_distance(self, word1, word2):
        """Levenshtein distance between two words"""
        if len(word1) < len(word2):
            return self.edit_distance(word2, word1)
        if len(word2) == 0:
            return len(word1)
        previous_row = range(len(word2) + 1)
        for i, c1 in enumerate(word1):
            current_row = [i + 1]
            for j, c2 in enumerate(word2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def autocorrect(self, word, max_suggestions=3):
        """Suggest corrections for a potentially misspelled word"""
        word = word.lower()
        if word in self.vocab:
            return [word]
        distances = [(w, self.edit_distance(word, w)) for w in self.vocab]
        distances.sort(key=lambda x: (x[1], -self.word_counts[x[0]]))
        return [w for w, d in distances[:max_suggestions] if d <= 2]

    def predict_next_word(self, prev_words, top_k=3):
        """Predict next likely word using n-gram model"""
        if not prev_words:
            return []
        prev_words = [word.lower() for word in prev_words[-self.n + 1:]]
        ngram = tuple(prev_words)
        candidates = self.ngram_model.get(ngram, {})
        if not candidates:
            candidates = self.word_counts
        return [word for word, _ in Counter(candidates).most_common(top_k)]

    def process_input(self, text):
        """Process input text, return autocorrected words and predictions"""
        words = word_tokenize(text.lower())
        corrected_words = []
        suggestions = []

        for i, word in enumerate(words):
            corrected = self.autocorrect(word)
            corrected_word = corrected[0] if corrected else word
            corrected_words.append(corrected_word)

            # Get context window for prediction
            prev_words = corrected_words[-(self.n - 1):] if i >= self.n - 2 else corrected_words
            next_words = self.predict_next_word(prev_words)
            suggestions.append((word, next_words))

        return corrected_words, suggestions

# Flask App Setup
app = Flask(__name__)
keyboard = AutocorrectKeyboard()

@app.route('/')
def index():
    return render_template('index.html')  # Make sure this HTML exists

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.get_json()
        user_input = data.get('text', '').strip()
        if not user_input:
            return jsonify({'error': 'Input cannot be empty'}), 400

        corrected, suggestions = keyboard.process_input(user_input)
        return jsonify({
            'corrected_text': ' '.join(corrected),
            'predictions': [{'word': word, 'next_words': preds} for word, preds in suggestions]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
