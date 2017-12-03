from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re

class CapsVectorizer(CountVectorizer):

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=False, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64):
        super(CapsVectorizer, self).__init__()

    def prepare_doc(self, doc):
        words = re.findall('\\b[A-Z,\']{4,}\\b', doc)
        print(words)
        doc = ' '.join(words)
        return doc

    def build_analyzer(self):
        preprocess = self.build_preprocessor()
        return lambda doc : preprocess(self.decode(self.prepare_doc(doc)))