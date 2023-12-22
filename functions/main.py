# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

# from urllib.parse import uses_query
from firebase_functions import https_fn
from firebase_admin import initialize_app

# import numpy as np
# import pandas as pd
# from spacy import nlp
# from nltk.stem import SnowballStemmer
# import spacy
# import json
# from bpemb import BPEmb
# import tensorflow as tf

from urllib.parse import uses_query
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from spacy.lang.id import Indonesian
from bpemb import BPEmb
import string
from spacy.lang.id.stop_words import STOP_WORDS
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity

nlp = Indonesian()
factory = StemmerFactory()
stemmer = factory.create_stemmer()



initialize_app()


@https_fn.on_request()
def on_request_example(req: https_fn.Request) -> https_fn.Response:
    return https_fn.Response("Hello world!")

def print_text(req: https_fn.Request) -> https_fn.Response:
    def print_text_in_df(doc):
        for row in range(0,doc.shape[0]):
            print(doc.iloc[row,0])
        return doc
    return https_fn.Response(print_text_in_df(req))

def word_token_spacy(req: https_fn.Request) -> https_fn.Response:
    def word_token_spacy(doc):
        doc1 = nlp(doc)
        token = [token.text for token in doc1]
        return token
    return https_fn.Response(word_token_spacy(req))

def char_token(req: https_fn.Request) -> https_fn.Response:
    def char_token(doc):
        token = [x for x in doc]
        return token
    return https_fn.Response(char_token(req))

def pre_token(req: https_fn.Request) -> https_fn.Response:
    def pre_token(doc):
        doc1 = doc.lower()
        doc2 = doc1.translate(str.maketrans('', '', string.punctuation + string.digits))
        doc3 = doc2.strip()
        return doc3
    return https_fn.Response(pre_token(req))

def pre_token_print(req: https_fn.Request) -> https_fn.Response:
    def pre_token_print(doc):
        doc1 = doc.lower()
        doc2 = doc1.translate(str.maketrans('', '', string.punctuation + string.digits))
        doc3 = doc2.strip()
        return doc3
    return https_fn.Response(pre_token_print(req))

def stopwords_removal(req: https_fn.Request) -> https_fn.Response:
    def stopwords_removal(words,stopword):
        return [word for word in words if word not in stopword]
    return https_fn.Response(stopwords_removal(req))

def find_word(req: https_fn.Request) -> https_fn.Response:
    def find_word(word,doc):
        return list(filter(lambda x: word in x, doc))
    return https_fn.Response(find_word(req))

def replace_slang_word(req: https_fn.Request) -> https_fn.Response:
    def replace_slang_word(doc,slang_word):
        for index in  range(0,len(doc)-1):
            index_slang = slang_word.slang==doc[index]
            formal = list(set(slang_word[index_slang].formal))
            if len(formal)==1:
                doc[index]=formal[0]
        return doc
    return https_fn.Response(replace_slang_word(req))

def lemma_indo(req: https_fn.Request) -> https_fn.Response:
    def lemma_indo(doc):
        return [stemmer.stem(word) for word in doc]
    return https_fn.Response(lemma_indo(req))

def combineword(req: https_fn.Request) -> https_fn.Response:
    def combineword(doc):
        articles = ["" for i in range(len(doc))]
        for i in range(len(doc)):
            waduh = (' '.join(doc[i]))
            articles[i] = waduh
        return(articles)
    return https_fn.Response(combineword(req))

def cosine_similarity(req: https_fn.Request) -> https_fn.Response:
    def cosine_similarity(doc1,doc2):
        return cosine_similarity(doc1,doc2)
    return https_fn.Response(cosine_similarity(req))

def combineword(req: https_fn.Request) -> https_fn.Response:
    def combineword(doc):
        articles = ["" for i in range(len(doc))]
        for i in range(len(doc)):
            waduh = (' '.join(doc[i]))
            articles[i] = waduh
        return(articles)
    return https_fn.Response(combineword(req))

def preprocess(req: https_fn.Request) -> https_fn.Response:
    def preprocess(dummy_doc):
        dummy_doc2 = dummy_doc.apply(lambda doc: word_token_spacy(doc.iloc[0]),axis=1)
        dummy_doc4 = dummy_doc.apply(lambda doc: word_token_spacy(doc.iloc[0]),axis=1)
        bpemb_id = BPEmb(lang="id",vs=200000, dim=300)
        dummy_doc5 = dummy_doc.apply(lambda doc: bpemb_id.encode(doc.iloc[0]),axis=1)
        dummy_doc6 = dummy_doc.apply(lambda doc: pre_token(doc.iloc[0]),axis=1)
        dummy_doc7 = dummy_doc.apply(lambda doc: stopwords_removal(word_token_spacy(doc.iloc[0]),STOP_WORDS),axis=1)
        indo_slang_word = pd.read_csv("https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv")
        dummy_doc9 = dummy_doc.apply(lambda doc: replace_slang_word(word_token_spacy(doc.iloc[0]),indo_slang_word),axis=1)
        dummy_doc10 = dummy_doc.apply(lambda doc: lemma_indo(word_token_spacy(doc.iloc[0])),axis=1)
        dummy_doc11 = combineword(dummy_doc10)
        return dummy_doc11
    return https_fn.Response(preprocess(req))
excel = pd.read_excel('advokat.xlsx', sheet_name='Sheet1')
kasuslengkap = excel[['Kasus Yang Diselesaikan']]
hasil_kasus = preprocess(kasuslengkap)

def load(req: https_fn.Request) -> https_fn.Response:
    def load(filename):	
        with open(filename) as data_file:
            data = json.load(data_file)	
    
        return data
    
    return https_fn.Response(load(req))

mydict = load('dict.json')

def getSinonim(req: https_fn.Request) -> https_fn.Response:
    def getSinonim(word):
        if word in mydict.keys():
            return mydict[word]['sinonim']
        else:
            return[]
    return https_fn.Response(getSinonim(req))

def gabung(req: https_fn.Request) -> https_fn.Response:
    def gabung(input_a):
        hasil_input = input_a.split()
        haha = []
        for i in range (len(hasil_input)):
            haha.append(getSinonim(hasil_input[i]))
        haha = [item for sublist in haha for item in sublist]
        haha.append(input_a)
        haha = ' '.join(haha)
        data = [[haha]]
        hasil = pd.DataFrame(data, columns=['kasus'])
        return hasil
    return https_fn.Response(gabung(req))

def search(req: https_fn.Request) -> https_fn.Response:
    def search(input):
        user_query = preprocess(gabung(input))[0]
    nlp = spacy.load("en_core_web_sm")
    processed_documents = [nlp(doc) for doc in hasil_kasus]  # Assuming 'hasil_kasus' is the correct variable name
    processed_query = nlp(uses_query)

    # Extract features using TF-IDF with TensorFlow
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc.text for doc in processed_documents])
    query_vector = vectorizer.transform([processed_query.text])

    # Convert sparse matrix to dense NumPy array
    tfidf_matrix = tfidf_matrix.toarray()
    query_vector = query_vector.toarray()

    # Calculate cosine similarity between the query and documents using TensorFlow
    cosine_similarities = tf.reduce_sum(tf.multiply(query_vector, tfidf_matrix), axis=1)
    cosine_similarities = cosine_similarities.numpy().flatten()

    # Get the index of the most similar document
    top_5_indices = np.argsort(cosine_similarities)[-5:][::-1]

    # Create a DataFrame with the search results
    search_results = pd.DataFrame({
        'Nama Pengacara': excel.loc[top_5_indices, 'Nama Pengacara'],
        'Cosine Similarity': cosine_similarities[top_5_indices]
    })

    # Display the result
    for index in top_5_indices:
        print(excel.loc[index, 'Nama Pengacara'])
        
    return https_fn.Response(search(req))


