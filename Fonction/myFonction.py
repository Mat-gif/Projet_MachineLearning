import warnings

from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", category=FutureWarning)

import re
import string

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import spacy

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')

import pandas as pd

def MyCleanText(X,
                lowercase=True, # mettre en minuscule
                removestopwords=True, # supprimer les stopwords
                removedigit=True, # supprimer les nombres
                getstemmer=True, # conserver la racine des termes
                getlemmatisation=True # lematisation des termes
                ):

    sentence=str(X)

    # suppression des caractères spéciaux
    sentence = re.sub(r'[^\w\s]',' ', sentence)
    # suppression de tous les caractères uniques
    sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)
    # substitution des espaces multiples par un seul espace
    sentence = re.sub(r'\s+', ' ', sentence, flags=re.I)

    # decoupage en mots
    tokens = word_tokenize(sentence)
    if lowercase:
        tokens = [token.lower() for token in tokens]

    # suppression ponctuation
    table = str.maketrans('', '', string.punctuation)
    words = [token.translate(table) for token in tokens]

    # suppression des tokens non alphabetique ou numerique
    words = [word for word in words if word.isalnum()]

    # suppression des tokens numerique
    if removedigit:
        words = [word for word in words if not word.isdigit()]

    # suppression des stopwords
    if removestopwords:
        words = [word for word in words if not word in stop_words]

    # lemmatisation
    if getlemmatisation:
        lemmatizer=WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word)for word in words]


    # racinisation
    if getstemmer:
        ps = PorterStemmer()
        words=[ps.stem(word) for word in words]

    sentence= ' '.join(words)

    return sentence


from sklearn.base import BaseEstimator, TransformerMixin

class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 removestopwords=False, # suppression des stopwords
                 lowercase=False,# passage en minuscule
                 removedigit=False, # supprimer les nombres
                 getstemmer=False,# racinisation des termes
                 getlemmatisation=False # lemmatisation des termes
                 ):

        self.lowercase=lowercase
        self.getstemmer=getstemmer
        self.removestopwords=removestopwords
        self.getlemmatisation=getlemmatisation
        self.removedigit=removedigit

    def transform(self, X, **transform_params):
        # Nettoyage du texte
        X=X.copy() # pour conserver le fichier d'origine
        return [MyCleanText(text,lowercase=self.lowercase,
                            getstemmer=self.getstemmer,
                            removestopwords=self.removestopwords,
                            getlemmatisation=self.getlemmatisation,
                            removedigit=self.removedigit) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)

    def get_params(self, deep=True):
        return {
            'lowercase':self.lowercase,
            'getstemmer':self.getstemmer,
            'removestopwords':self.removestopwords,
            'getlemmatisation':self.getlemmatisation,
            'removedigit':self.removedigit
        }

    def set_params (self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter,value)
        return self

def balanceSample(X, size, classe):
    data = pd.DataFrame()
    for i in classe:
        x = X.loc[X['our rating'].isin([i])]
        if( x.shape[0] > size) :
            x_new =  pd.concat([x[x['our rating'] == c].sample(size, replace=False)for c in x['our rating'].unique()])
        elif (x.shape[0] < size) :
            if(x.shape[0]*2 >= size):
                x_new = pd.concat([x,pd.concat([x[x['our rating'] == c].sample((size-x.shape[0]), replace=False)for c in x['our rating'].unique()])], ignore_index = True)
            else :
                x_new = pd.concat([x,pd.concat([x[x['our rating'] == c].sample((size-x.shape[0]), replace=True)for c in x['our rating'].unique()])], ignore_index = True)
        else :
            x_new = x
        data = pd.concat([data,x_new], ignore_index = True)
    #print(data['our rating'].value_counts())

    return data



from sklearn.decomposition import NMF


def extractTopic(mySample):
    mySample['topic']= mySample['text']
    # documents = data.iloc[:, 1:].values
    # documents = np.char.lower(documents.astype('U'))
    documents = mySample['text'].values.astype('U')

    # Vectorisation des données textuelles

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    # X

    # Application de la NMF sur la matrice de données vectorielles
    model = NMF(n_components=10, init='random', random_state=0)
    W = model.fit_transform(X)
    H = model.components_

    # # Affichage des sujets (ou des thèmes)
    for topic_idx, topic in enumerate(H):
        mySample['topic'][topic_idx] = " ".join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-10 - 1:-1]])

    return mySample