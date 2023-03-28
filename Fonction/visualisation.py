from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

from Fonction.myFonction import TextNormalizer
import plotly.express as px

def myAcP_2D_3D(X,y,lemma=False,digit=False,stopwords=None):
    text_normalizer=TextNormalizer(getlemmatisation=lemma, removedigit=digit)

    # appliquer fit.transform pour réaliser les pré-traitements sur X
    X_cleaned=text_normalizer.fit_transform(X)


    tfidf=TfidfVectorizer(stop_words=stopwords) #max_features=10

    vector_tfidf=tfidf.fit_transform(X_cleaned)



    from sklearn.decomposition import TruncatedSVD

    #2D
    svd=TruncatedSVD(n_components=2, random_state=0)
    components = svd.fit_transform(vector_tfidf)
    fig = px.scatter(components, x=0, y=1, color=y)
    fig.show()
    #3D
    svd=TruncatedSVD(n_components=3, random_state=0)
    components = svd.fit_transform(vector_tfidf)
    fig = px.scatter_3d(
        components, x=0, y=1, z=2, color=y,
        title='TruncatedSVD',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
    )
    fig.show()


def myTSNE_2d_3d(X,y,lemma=False,digit=False,stopwords=None):
    text_normalizer=TextNormalizer(getlemmatisation=lemma, removedigit=digit)

        # appliquer fit.transform pour réaliser les pré-traitements sur X
    X_cleaned=text_normalizer.fit_transform(X)


    tfidf=TfidfVectorizer(stop_words=stopwords) #max_features=10

    vector_tfidf=tfidf.fit_transform(X_cleaned)


    tsne = TSNE(n_components=2, random_state=0)
    projections = tsne.fit_transform(vector_tfidf.toarray())
    fig = px.scatter(
        projections, x=0, y=1,
        color=y, labels={'color': 'our rating'}
    )
    fig.show()
    # 3D
    tsne = TSNE(n_components=3, random_state=0)
    projections = tsne.fit_transform(vector_tfidf.toarray())

    fig = px.scatter_3d(
        projections, x=0, y=1, z=2,
        color=y, labels={'color': 'our rating'}
    )
    fig.update_traces(marker_size=5)
    fig.show()


from umap import UMAP

def myUMAP_2d_3d(X,y,lemma=False,digit=False,stopwords=None):
    text_normalizer=TextNormalizer(getlemmatisation=lemma, removedigit=digit)

    # appliquer fit.transform pour réaliser les pré-traitements sur X
    X_cleaned=text_normalizer.fit_transform(X)


    tfidf=TfidfVectorizer(stop_words=stopwords) #max_features=10

    vector_tfidf=tfidf.fit_transform(X_cleaned)

    umap = UMAP(n_components=2, init='random', random_state=0)
    projection = umap.fit_transform(vector_tfidf.toarray())
    fig = px.scatter(
        projection, x=0, y=1,
        color=y, labels={'color': 'our rating'}
    )
    fig.show()
    # 3D
    umap = UMAP(n_components=3, init='random', random_state=0)
    projection = umap.fit_transform(vector_tfidf.toarray())
    fig = px.scatter_3d(
        projection, x=0, y=1,z=2,
        color=y, labels={'color': 'our rating'}
    )
    fig.update_traces(marker_size=3)
    fig.show()