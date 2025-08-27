# sentence_builder.py

import time

class SentenceBuilder:
    def __init__(self, max_words=10, min_conf=0.8, cooldown=1.5):
        """
        max_words : maximum words to keep in the sentence
        min_conf  : minimum confidence to accept a prediction
        cooldown  : seconds to wait before allowing the same word again
        """
        self.words = []
        self.max_words = max_words
        self.min_conf = min_conf
        self.cooldown = cooldown
        self.last_added = {}  # track last time each word was added

    def add_word(self, word, confidence):
        now = time.time()

        # filter out low confidence
        '''if confidence < self.min_conf:
            return'''  

        # prevent adding same word too quickly
        if word in self.last_added and (now - self.last_added[word]) < self.cooldown:
            return  

        # add word
        self.words.append(word)
        self.last_added[word] = now

        # keep only last N words
        if len(self.words) > self.max_words:
            self.words.pop(0)

    def get_sentence(self):
        return " ".join(self.words)
'''from collections import deque, Counter

class SentenceBuilder:
    def __init__(self, window_size=15, min_repeats=8):
        self.words = []
        self.history = deque(maxlen=window_size)
        self.min_repeats = min_repeats

    def add_word(self, word, conf):
        self.history.append(word)
        most_common, count = Counter(self.history).most_common(1)[0]
        if count >= self.min_repeats:
            if len(self.words) == 0 or self.words[-1] != most_common:
                self.words.append(most_common)

    def get_sentence(self):
        return " ".join(self.words)'''

'''class SentenceBuilder:
    def __init__(self, cooldown_frames=10, confidence_threshold=0.6):
        self.sentence = []
        self.prev_word = None
        self.cooldown = 0
        self.cooldown_frames = cooldown_frames
        self.conf_threshold = confidence_threshold

    def add_word(self, word, confidence):
        if self.cooldown == 0 and confidence >= self.conf_threshold:
            if not self.sentence or self.sentence[-1] != word:
                self.sentence.append(word)
                self.prev_word = word
                self.cooldown = self.cooldown_frames
        if self.cooldown > 0:
            self.cooldown -= 1

    def get_sentence(self):
        return " ".join(self.sentence)

    def clear_last(self):
        if self.sentence:
            self.sentence.pop()

    def clear_all(self):
        self.sentence = []'''
