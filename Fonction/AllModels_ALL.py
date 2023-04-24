import time

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from Fonction.MyNLPUtilities import Result
from Fonction.myFonction import TextNormalizer

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def testAllModel(X_train,y_train,n):
    # creation du tableau des différents classifieur
    models = []
    models.append(('MultinomialNB',MultinomialNB())) ##
    models.append(('LR', LogisticRegression(solver='lbfgs'))) ##
    models.append(('KNN', KNeighborsClassifier())) ##
    models.append(('CART', DecisionTreeClassifier())) ##
    models.append(('RF', RandomForestClassifier())) ##
    models.append(('SVM', SVC())) ##

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
    import nltk
    nltk.download('omw-1.4')
    #import warnings
    #warnings.filterwarnings('ignore')
    nbRep = n


    seed = 7
    allresults = []
    results = []
    names = []

    score = 'accuracy'
    # Nous appliquons les pré-traitements sur X
    # Nous appliquons les pré-traitements sur X

    text_normalizer=TextNormalizer()
    # appliquer fit.transform pour réaliser les pré-traitements sur X
    X_cleaned=text_normalizer.fit_transform(X_train)

    # pour l'enchainer avec un tf-idf et obtenir une matrice
    tfidf=TfidfVectorizer()
    features=tfidf.fit_transform(X_cleaned).toarray()

    # attention ici il faut passer features dans cross_val_score plutôt que X

    for name,model in models:
        # cross validation en 10 fois
        kfold = KFold(n_splits=10, random_state=seed, shuffle=True)

        print ("Evaluation de ",name)
        start_time = time.time()
        # application de la classification
        cv_results = cross_val_score(model, features, y_train, cv=kfold, scoring=score)

        # pour afficher les paramètres du modèle en cours et la taille du vecteur intermédiaire
        # enlever le commentaire des deux lignes suivantes
        #print ("paramètre du modèle ",model.get_params(),'\n')
        #print ("taille du vecteur : ",(model.named_steps['tfidf_vectorizer'].fit_transform(X)).shape,'\n')

        thetime=time.time() - start_time
        result=Result(name,cv_results.mean(),cv_results.std(),thetime)
        allresults.append(result)
        # pour affichage
        results.append(cv_results)
        names.append(name)
        print("%s : %0.3f (%0.3f) in %0.3f s" % (name, cv_results.mean(), cv_results.std(),thetime))

    allresults=sorted(allresults, key=lambda result: result.scoremean, reverse=True)

    # affichage des résultats
    print ('\nLe meilleur resultat : ')
    print ('Classifier : ',allresults[0].name,
           ' %s : %0.3f' %(score,allresults[0].scoremean),
           ' (%0.3f)'%allresults[0].stdresult,
           ' en %0.3f '%allresults[0].timespent,' s\n')

    print ('Tous les résultats : \n')
    for result in allresults:
        print ('Classifier : ',result.name,
               ' %s : %0.3f' %(score,result.scoremean),
               ' (%0.3f)'%result.stdresult,
               ' en %0.3f '%result.timespent,' s')

    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.suptitle('Comparaison des algorithmes')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)



from sklearn.model_selection import GridSearchCV





def testSVC(X_train,y_train,n):

    pipeline=Pipeline([("cleaner", TextNormalizer()),
                       ("tfidf", TfidfVectorizer()),
                       ('svm', SVC())])

    # creation des différents paramètres à tester pour SVM
    # Attention dans le pipeline le nom pour le classifier SVM est : svm même si l'algorithme s'appelle SVC
    # pour le référencer il faut utiliser le nom utilisé, i.e. svm, puis deux caractères soulignés
    # et enfin le nom du paramètre
    parameters = {
        'cleaner__getstemmer':[True,False],
        'cleaner__removedigit': [True,False],
        'cleaner__getlemmatisation': [True,False],
        'tfidf__stop_words':['english',None],
        'tfidf__lowercase': [True,False],
        'svm__C': [0.001, 0.01, 0.1, 1, 10],
        'svm__gamma' : [0.001, 0.01, 0.1, 1],
        'svm__kernel': ['linear','rbf','poly','sigmoid']
    }


    score='accuracy'

    # Application de gridsearchcv, n_jobs=-1 permet de pouvoir utiliser plusieurs CPU s'ils sont disponibles
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1,scoring=score,cv=n)

    print("Application de gridsearch ...")
    print("pipeline :", [name for name, _ in pipeline.steps])
    print("parameters :")
    print(parameters)
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    print("réalisé en  %0.3f s" % (time.time() - start_time))
    print("Meilleur résultat : %0.3f" % grid_search.best_score_)

    # autres mesures et matrice de confusion
    # y_pred = grid_search.predict(X_test)
    # MyshowAllScores(y_test,y_pred)


    print("Ensemble des meilleurs paramètres :")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # Affichage des premiers résultats du gridsearch
    df_results=pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),
                          pd.DataFrame(grid_search.cv_results_["mean_test_score"],
                                       columns=[score])],axis=1).sort_values(score,ascending=False)
    print ("\nLes premiers résultats : \n",df_results.head())


def testRFC(X_train,y_train,n):


    pipeline=Pipeline([("cleaner", TextNormalizer()),
                       ("tfidf", TfidfVectorizer()),
                       ('rfc', RandomForestClassifier())
                       ]
                      )


    parameters = {
        'cleaner__getstemmer':[True,False],
        'cleaner__removedigit':[True,False],
        'cleaner__getlemmatisation':[True,False],
        'tfidf__stop_words':['english',None],
        'tfidf__lowercase': [True,False],
        'rfc__n_estimators': [500, 1200],
        'rfc__max_depth': [25, 30],
        'rfc__min_samples_split': [5, 10, 15],
        'rfc__min_samples_leaf' : [1, 2]
    }


    score='accuracy'
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,  verbose=1,scoring=score,cv=n)

    print("Application de gridsearch ...")
    print("pipeline :", [name for name, _ in pipeline.steps])
    print("parameters :")
    print(parameters)
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    print("réalisé en  %0.3f s" % (time.time() - start_time))
    print("Meilleur résultat : %0.3f" % grid_search.best_score_)

    # matrice de confusion
    #y_pred = grid_search.predict(X_test)
    #MyshowAllScores(y_test,y_pred)

    print("Ensemble des meilleurs paramètres :")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # Affichage des premiers résultats du gridsearch
    df_results=pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),
                          pd.DataFrame(grid_search.cv_results_["mean_test_score"],
                                       columns=[score])],axis=1).sort_values(score,ascending=False)
    print ("\nLes premiers résultats : \n",df_results.head())



def testLR(X_train,y_train,n):


    pipeline=Pipeline([("cleaner", TextNormalizer()),
                       ("tfidf", TfidfVectorizer()),
                       ("lr", LogisticRegression()),
                       ]
                      )


    parameters = {
        'cleaner__getstemmer':[True,False],
        'cleaner__removedigit':[True,False],
        'cleaner__getlemmatisation':[True,False],
        'tfidf__stop_words':['english',None],
        'tfidf__lowercase': [True,False],
        'lr__solver' : ['newton-cg', 'lbfgs', 'liblinear'],
        'lr__penalty' : ['l2'],
        'lr__C' : [100, 10, 1.0, 0.1, 0.01]
    }


    score='accuracy'
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,  verbose=1,scoring=score,cv=n)

    print("Application de gridsearch ...")
    print("pipeline :", [name for name, _ in pipeline.steps])
    print("parameters :")
    print(parameters)
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    print("réalisé en  %0.3f s" % (time.time() - start_time))
    print("Meilleur résultat : %0.3f" % grid_search.best_score_)

    # matrice de confusion
    #y_pred = grid_search.predict(X_test)
    #MyshowAllScores(y_test,y_pred)

    print("Ensemble des meilleurs paramètres :")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # Affichage des premiers résultats du gridsearch
    df_results=pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),
                          pd.DataFrame(grid_search.cv_results_["mean_test_score"],
                                       columns=[score])],axis=1).sort_values(score,ascending=False)
    print ("\nLes premiers résultats : \n",df_results.head())
    
def testKNeighborsClassifier(X_train,y_train,n):


    pipeline=Pipeline([("cleaner", TextNormalizer()),
                       ("tfidf", TfidfVectorizer()),
                       ("KNN", KNeighborsClassifier()),
                       ]
                      )


    parameters = {
        'cleaner__getstemmer':[True,False],
        'cleaner__removedigit':[True,False],
        'cleaner__getlemmatisation':[True,False],
        'tfidf__stop_words':['english',None],
        'tfidf__lowercase': [True,False],
        'KNN__n_neighbors': list(range(1,15)),
        'KNN__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'KNN__weights': ['uniform', 'distance'],
        'KNN__metric': ['minkowski','euclidean','manhattan']
    }


    score='accuracy'
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,  verbose=1,scoring=score,cv=n)

    print("Application de gridsearch ...")
    print("pipeline :", [name for name, _ in pipeline.steps])
    print("parameters :")
    print(parameters)
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    print("réalisé en  %0.3f s" % (time.time() - start_time))
    print("Meilleur résultat : %0.3f" % grid_search.best_score_)

    # matrice de confusion
    #y_pred = grid_search.predict(X_test)
    #MyshowAllScores(y_test,y_pred)

    print("Ensemble des meilleurs paramètres :")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # Affichage des premiers résultats du gridsearch
    df_results=pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),
                          pd.DataFrame(grid_search.cv_results_["mean_test_score"],
                                       columns=[score])],axis=1).sort_values(score,ascending=False)
    print ("\nLes premiers résultats : \n",df_results.head())
    
def testCART(X_train,y_train,n):

    pipeline=Pipeline([("cleaner", TextNormalizer()),
                       ("tfidf", TfidfVectorizer()),
                       ("CART", DecisionTreeClassifier()),
                       ]
                      )


    parameters = {
        'cleaner__getstemmer':[True,False],
        'cleaner__removedigit':[True,False],
        'cleaner__getlemmatisation':[True,False],
        'tfidf__stop_words':['english',None],
        'tfidf__lowercase': [True,False],
        'CART__max_depth': [10, 20, 30],
        'CART__min_samples_split': [2, 5, 10],
        'CART__min_samples_leaf': [1, 2, 4],
        'CART__criterion': ['gini', 'entropy']
    }


    score='accuracy'
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,  verbose=1,scoring=score,cv=n)

    print("Application de gridsearch ...")
    print("pipeline :", [name for name, _ in pipeline.steps])
    print("parameters :")
    print(parameters)
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    print("réalisé en  %0.3f s" % (time.time() - start_time))
    print("Meilleur résultat : %0.3f" % grid_search.best_score_)

    # matrice de confusion
    #y_pred = grid_search.predict(X_test)
    #MyshowAllScores(y_test,y_pred)

    print("Ensemble des meilleurs paramètres :")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # Affichage des premiers résultats du gridsearch
    df_results=pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),
                          pd.DataFrame(grid_search.cv_results_["mean_test_score"],
                                       columns=[score])],axis=1).sort_values(score,ascending=False)
    print ("\nLes premiers résultats : \n",df_results.head())
    
    
def testMultinomialNB(X_train,y_train,n):
    pipeline=Pipeline([("cleaner", TextNormalizer()),
                       ("tfidf", TfidfVectorizer()),
                       ("MultinomialNB", MultinomialNB()),
                       ]
                      )


    parameters = {
        'cleaner__getstemmer':[True,False],
        'cleaner__removedigit':[True,False],
        'cleaner__getlemmatisation':[True,False],
        'tfidf__stop_words':['english',None],
        'tfidf__lowercase': [True,False],
        'MultinomialNB__alpha': [0.1, 0.5, 1.0, 2.0],
        'MultinomialNB__fit_prior':[True, False],
        'MultinomialNB__force_alpha':[True, False]
    }


    score='accuracy'
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,  verbose=1,scoring=score,cv=n)

    print("Application de gridsearch ...")
    print("pipeline :", [name for name, _ in pipeline.steps])
    print("parameters :")
    print(parameters)
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    print("réalisé en  %0.3f s" % (time.time() - start_time))
    print("Meilleur résultat : %0.3f" % grid_search.best_score_)

    # matrice de confusion
    #y_pred = grid_search.predict(X_test)
    #MyshowAllScores(y_test,y_pred)

    print("Ensemble des meilleurs paramètres :")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # Affichage des premiers résultats du gridsearch
    df_results=pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),
                          pd.DataFrame(grid_search.cv_results_["mean_test_score"],
                                       columns=[score])],axis=1).sort_values(score,ascending=False)
    print ("\nLes premiers résultats : \n",df_results.head())