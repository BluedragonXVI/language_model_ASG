import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from streamlit.report_thread import get_report_ctx
import config

# Text clustering libraries:
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

# get a unique session ID that can used at postgres primary key 
def get_session_id() -> str:
    session_id = get_report_ctx().session_id
    session_id = session_id.replace('-','_')
    session_id = '_id_' + session_id # postgres name convention
    return session_id

# functions to read/write states of user input and dataframes
def write_state(column, value, engine, session_id):
    engine.execute("UPDATE %s SET %s='%s'" % (session_id, column, value))

def write_state_df(df:pd.DataFrame, engine, session_id):
    df.to_sql('%s' % (session_id),engine,index=False,if_exists='replace',chunksize=1000)

def read_state(column, engine, session_id):
    state_var = engine.execute("SELECT %s FROM %s" % (column, session_id))
    state_var = state_var.first()[0]
    return state_var

def read_state_df(engine, session_id):
    try:
        df = pd.read_sql_table(session_id, con=engine)
    except:
        df = pd.DataFrame([])
    return df

if __name__ == '__main__':

    # create PostgreSQL client using configuration file 
    engine = create_engine
    # End of neural SLAM code

    # All code below is from text clustering school project with sklearn of newsgroups data to test working webapp and streamlit plotting 
    # Load newsgroup data and descriptive information

    categories = ['sci.space', 'rec.sport.baseball', 'sci.med', 'rec.autos', 'misc.forsale'] # set desired categories here
    train_data = fetch_20newsgroups(subset='train', categories=categories)
    count_cat = Counter(train_data.target)
    num_docs = 0 # will be updated when items are counted next
    for cat, size in count_cat.items():
        print("Category:", train_data.target_names[cat], "Size:", size)
        num_docs += size
    print(train_data.filenames.shape) # 2963 of all 5 categories
    print(train_data.target.shape) # 2963 of all 5 categories

    """ Selected categories and their size:

    Category: rec.sport.baseball Size: 597
    Category: sci.med Size: 594
    Category: rec.autos Size: 594
    Category: sci.space Size: 593
    Category: misc.forsale Size: 585

    """

    num_features = 5000
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=num_features, stop_words='english', use_idf=True)
    train_vector = vectorizer.fit_transform(train_data.data)
    print("number_of_samples: %d, number_of_features: %d" % train_vector.shape)
    # number_of_samples: 1190, number_of_features: 1000 
    # original number of features: 21317

    # Create the TF_IDF vectors, cluster and get silhouette coefficient
    num_clusters = 5
    tf_cluster_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, batch_size=5000, n_init=3, verbose=0)
    tf_cluster_model.fit(train_vector)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(train_vector, tf_cluster_model.labels_, sample_size=1000))
    # Silhouette Coefficient: 0.009

    order_centroids = tf_cluster_model.cluster_centers_.argsort()[:, ::-1]  # sort and reverse
    terms = vectorizer.get_feature_names() # feature names from tf_idf vector

    # print first ten terms from each cluster
    for i in range(num_clusters):
        print("\nTop 10 terms in cluster %d:" % i) #replace categories[i] with tf_cluster_model names
        for ind in order_centroids[i, :10]:  
            print(' %s' % terms[ind])

    # Compute homogeneity with newsgroups labels
    train_labels = train_data.target
    cluster_labels = tf_cluster_model.labels_
    print("\nHomogeneity: %0.3f" % metrics.homogeneity_score(train_labels, cluster_labels))

    # Compute purity of the clusters (space, baseball, etc...)
    # Create the confusion matrix
    confusion_matrix = metrics.cluster.contingency_matrix(train_labels, cluster_labels)
    purity = np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix)
    count_clusters = Counter(cluster_labels)
    print("Purity: ", purity, "\nNumber of features: ", num_features, "\nNumber of clusters: ", num_clusters)
    for cluster, size in count_clusters.items():
        print("Cluster:", cluster, "Size:", size)
    
    # SVD Dimensionality Reduction
    """As described in: https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py:
    Vectorizer results are normalized, which makes KMeans behave as
    spherical k-means for better results. Since LSA/SVD results are
    not normalized, we have to redo the normalization."""

    svd = TruncatedSVD(75)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    svd_vector = lsa.fit_transform(train_vector)

    km = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1, init_size=1000, batch_size=1000)
    km.fit(svd_vector)

    # Reupdate the cluster labels
    cluster_labels = km.labels_

    # Compute purity as in the tf_idf above
    # Create the confusion matrix
    confusion_matrix = metrics.cluster.contingency_matrix(train_labels, cluster_labels)
    purity = np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix)
    count_clusters = Counter(cluster_labels)
    print("Purity: ", purity, "\nNumber of features: ", num_features, "\nNumber of clusters: ", num_clusters)
    for cluster, size in count_clusters.items():
        print("Cluster:", cluster, "Size:", size)

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

