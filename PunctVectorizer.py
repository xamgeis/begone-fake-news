from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import string

class PunctVectorizer(CountVectorizer):

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64):
        super(PunctVectorizer, self).__init__()

    def prepare_doc(self, doc):
        punc_list = string.punctuation
        doc = doc.replace("\\r\\n"," ")
        for character in doc:
            if character not in punc_list:
                doc = doc.replace(character, "")
        return doc

    def build_analyzer(self):
        preprocess = self.build_preprocessor()
        return lambda doc : preprocess(self.decode(self.prepare_doc(doc)))