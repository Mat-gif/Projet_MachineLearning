from gettext import install

import learn as learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
# Umap
#!pip install umap-learn[plot]


from Fonction.myFonction import TextNormalizer
import plotly.express as px

def myAcP_2D_3D(X,y,lemma=True,digit=True):
    text_normalizer=TextNormalizer(getlemmatisation=lemma, removedigit=digit, removestopwords=True)

    # appliquer fit.transform pour réaliser les pré-traitements sur X
    X_cleaned=text_normalizer.fit_transform(X)


    tfidf=TfidfVectorizer(lowercase=False) #max_features=10

    vector_tfidf=tfidf.fit_transform(X_cleaned)



    from sklearn.decomposition import TruncatedSVD

    #2D
    svd=TruncatedSVD(n_components=3, random_state=0)
    components = svd.fit_transform(vector_tfidf)

    fig = px.scatter(components, x=0, y=1, color=y)

    # Get the explained variance ratios for the two components
    variance_ratios = svd.explained_variance_ratio_

    # Add annotations to show the percentage of variance explained by each component
    fig.add_annotation(x=0.05, y=0.95,
                       text=f"Component 1: {variance_ratios[0]*100:.2f}%",
                       showarrow=False, yshift=10)
    fig.add_annotation(x=0.95, y=0.05,
                       text=f"Component 2: {variance_ratios[1]*100:.2f}%",
                       showarrow=False, yshift=-10)

    fig.show()

    fig = px.scatter(components, x=0, y=2, color=y)

    fig.show()
    fig = px.scatter(components, x=1, y=2, color=y)

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


import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.graph_objs as go
def myTSNE_2d_3d(X,y,lemma=True,digit=True):
    text_normalizer=TextNormalizer(getlemmatisation=lemma, removedigit=digit, removestopwords=True)

    # appliquer fit.transform pour réaliser les pré-traitements sur X
    X_cleaned=text_normalizer.fit_transform(X)


    tfidf=TfidfVectorizer(lowercase=False) #max_features=10


    vector_tfidf=tfidf.fit_transform(X_cleaned)


    tsne = TSNE(n_components=2, random_state=0)
    projections = tsne.fit_transform(vector_tfidf.toarray())

    # calculate the centroids of each cluster
    centroids = {}
    for i in range(len(y)):
        if y[i] not in centroids:
            centroids[y[i]] = [projections[i]]
        else:
            centroids[y[i]].append(projections[i])

    # plot the scatter plot with centroids
    fig = px.scatter(
        projections, x=0, y=1,
        color=y, labels={'color': 'our rating'}
    )
    for c in centroids:
        x = [p[0] for p in centroids[c]]
        y = [p[1] for p in centroids[c]]
        fig.add_trace(go.Scatter(x=[sum(x)/len(x)], y=[sum(y)/len(y)],
                                 mode='markers',marker=dict(size=10),
                                 name="Centroid of "+str(c)))
    fig.show()





def myUMAP_2d_3d(X,y,lemma=True,digit=True):
    text_normalizer=TextNormalizer(getlemmatisation=lemma, removedigit=digit, removestopwords=True)

    # appliquer fit.transform pour réaliser les pré-traitements sur X
    X_cleaned=text_normalizer.fit_transform(X)


    tfidf=TfidfVectorizer(lowercase=False) #max_features=10


    vector_tfidf=tfidf.fit_transform(X_cleaned)

    from umap import UMAP
    umap = UMAP(n_neighbors=2, n_components=2, init='random', random_state=0)

    projection = umap.fit_transform(vector_tfidf)
    fig = px.scatter(
        projection, x=0, y=1,
        color=y, labels={'color': 'our rating'}
    )

    fig.show()
