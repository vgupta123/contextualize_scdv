###########################################
# Major portion of this script has been taken from https://github.com/dheeraj7596/SCDV
# Contains the main logic for SCDV
###########################################

import pickle 
import pandas as pd 
from sklearn.utils import shuffle
from gensim.models import Word2Vec, FastText
import pandas as pd
import time
import numpy as np
import sys
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import re
from nltk.corpus import stopwords


with open("df_contextualized.pkl", 'rb') as f:
    df = pickle.load(f)

with open('word_cluster_map.pkl', 'rb') as f:
    word_cluster = pickle.load(f)

embedding = []
key = []
model={}
for keys in word_cluster:
  if len(word_cluster[keys])==1:
    key.append(keys)
    embedding.append(word_cluster[keys][0])
    model[keys]=word_cluster[keys][0]
  else:
    print("##############")
    for i in range(len(word_cluster[keys])):
      key.append(keys+'$'+str(i))
      embedding.append(word_cluster[keys][i])
      model[keys+'$'+str(i)]=word_cluster[keys][i]

 
 with open('key.pkl', 'wb') as f:
  pickle.dump(key, f)

with open('embed.pkl', 'wb') as f:
  pickle.dump(embedding, f)


df = shuffle(df)
df = df.reset_index(drop=True)
df_train = df.truncate(before=0,after=0.7*(len(df)))
df_test = df.truncate(before=0.7*(len(df)))

df.to_csv("all_v2.tsv",sep='\t')
df_train.to_csv("train_v2.tsv",sep='\t')
df_test.to_csv("test_v2.tsv",sep='\t')


def review_to_wordlist(review, remove_stopwords=False):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        # 1. Remove non-alphabets 
        #review = re.sub("[^a-zA-Z]", " ", review)
        # 2. Convert words to lower case and split them
        words = review.split()
        # 3. Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops and len(w) > 1]
        # 4. Return a list of words
        return (words)

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

def cluster_GMM(num_clusters, word_vectors):

    # Initalize a GMM object and use it for clustering.
    clf = GaussianMixture(n_components=num_clusters,
                          covariance_type="full", init_params='kmeans', max_iter=90,verbose=3)
    # Get cluster assignments.
    print(len(word_vectors))
    clf.fit(word_vectors)
    idx = clf.predict(word_vectors)
    print("Clustering Done...", time.time() - start, "seconds")
    # Get probabilities of cluster assignments.
    idx_proba = clf.predict_proba(word_vectors)
    # Dump cluster assignments and probability of cluster assignments. 
    joblib.dump(idx, 'gmm_latestclusmodel_len2alldata.pkl')
    print("Cluster Assignments Saved...")

    joblib.dump(idx_proba, 'gmm_prob_latestclusmodel_len2alldata.pkl')
    print("Probabilities of Cluster Assignments Saved...")
    return (idx, idx_proba)

def read_GMM(idx_name, idx_proba_name):
    # Loads cluster assignments and probability of cluster assignments. 
    idx = joblib.load(idx_name)
    idx_proba = joblib.load(idx_proba_name)
    print("Cluster Model Loaded...")
    return (idx, idx_proba)


def get_probability_word_vectors(featurenames, word_centroid_map, num_clusters, word_idf_dict):
    # This function computes probability word-cluster vectors.

    prob_wordvecs = {}
    count = 0 
    exp=0

    for word in word_centroid_map:
        prob_wordvecs[word] = np.zeros(num_clusters * num_features, dtype="float32")
        for index in range(0, num_clusters):
            #prob_wordvecs[word][index * num_features:(index + 1) * num_features] = model[word] * word_centroid_prob_map[word][index] * word_idf_dict[word]
            count+=1
            try:
                prob_wordvecs[word][index * num_features:(index + 1) * num_features] = model[word] * word_centroid_prob_map[word][index] * word_idf_dict[word]
                #print(word)
            except:
                exp+=1
                continue
    print(exp/count)

    return prob_wordvecs

def create_cluster_vector_and_gwbowv(prob_wordvecs, wordlist, word_centroid_map, word_centroid_prob_map, dimension,
                                     word_idf_dict, featurenames, num_centroids, train=False):
    # This function computes SDV feature vectors.
    bag_of_centroids = np.zeros(num_centroids * dimension, dtype="float32")
    global min_no
    global max_no
    count = 0 
    exp_counter = 0

    for word in wordlist:
        count+=1
        try:
            temp = word_centroid_map[word]
        except:
            exp_counter+=1
            continue

        bag_of_centroids += prob_wordvecs[word]
    #print(exp_counter/count)
  

    norm = np.sqrt(np.einsum('...i,...i', bag_of_centroids, bag_of_centroids))
    if (norm != 0):
        bag_of_centroids /= norm

    # To make feature vector sparse, make note of minimum and maximum values.
    if train:
        min_no += min(bag_of_centroids)
        max_no += max(bag_of_centroids)

    return bag_of_centroids

if __name__ == '__main__':

    start = time.time()

    num_features = 768  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 40  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    #model_type = sys.argv[3]

    model_name = str(num_features) + "features_" + str(min_word_count) + "minwords_" + str(
        context) + "context_len2alldata"

    # Load train data.
    train = pd.read_csv('train_v2.tsv', header=0, delimiter="\t")
    # Load test data.
    test = pd.read_csv('test_v2.tsv', header=0, delimiter="\t")
    all = pd.read_csv('all_v2.tsv', header=0, delimiter="\t")

    import pickle 
    with open('key.pkl', 'rb') as f:
      index2word = pickle.load(f)
    with open('embed.pkl', 'rb') as f:
      word_vectors = pickle.load(f)
    
    num_clusters = 30
    idx, idx_proba = cluster_GMM(num_clusters, word_vectors)
    word_centroid_map = dict(zip(index2word, idx))
    word_centroid_prob_map = dict(zip(index2word, idx_proba))
    traindata = []
    
    for i in range(0, len(all["news"])):
        traindata.append(" ".join(review_to_wordlist(all["news"][i], True)))

    tfv = TfidfVectorizer(strip_accents='unicode', dtype=np.float32)
    tfidfmatrix_traindata = tfv.fit_transform(traindata)
    featurenames = tfv.get_feature_names()
    idf = tfv._tfidf.idf_

    def calculate_df_doc_freq(df):
      docfreq = {}
      docfreq["UNK"] = len(df)
      for index, row in df.iterrows():
          line = row["news"]
          words = line.strip().split()
          temp_set = list(set(words))
          for w in temp_set:
              try:
                  docfreq[w] += 1
              except:
                  docfreq[w] = 1
      return docfreq

    def calculate_inv_doc_freq(df, docfreq):
      inv_docfreq = {}
      N = len(df)
      print(N)
      for word in docfreq:
          inv_docfreq[word] = np.log(N / docfreq[word])
      return inv_docfreq

    # Creating a dictionary with word mapped to its idf value 
    print("Creating word-idf dictionary for Training set...")

    word_idf_dict = {}

    word_idf_dict = calculate_inv_doc_freq(all,calculate_df_doc_freq(all))

    print("Creating word-idf dictionary for Training set...")

    prob_wordvecs = get_probability_word_vectors(featurenames, word_centroid_map, num_clusters, word_idf_dict)

    temp_time = time.time() - start
    print("Creating Document Vectors...:", temp_time, "seconds.")

    gwbowv = np.zeros((train["news"].size, num_clusters * (num_features)), dtype="float32")

    counter = 0

    min_no = 0
    max_no = 0
    for review in train["news"]:
        words = review_to_wordlist(review, \
                                                         remove_stopwords=True)
        gwbowv[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, words, word_centroid_map,
                                                           word_centroid_prob_map, num_features, word_idf_dict,
                                                           featurenames, num_clusters, train=True)
        counter += 1
        if counter % 100 == 0:
            print("Train News Covered : ", counter)

    gwbowv_name = "SDV_" + str(num_clusters) + "cluster_" + str(num_features) + "feature_matrix_gmm_sparse.npy"

    gwbowv_test = np.zeros((test["news"].size, num_clusters * (num_features)), dtype="float32")

    counter = 0

    for review in test["news"]:
        words = review_to_wordlist(review, \
                                                         remove_stopwords=True)
        gwbowv_test[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, words, word_centroid_map,
                                                                word_centroid_prob_map, num_features, word_idf_dict,
                                                                featurenames, num_clusters)
        counter += 1
        if counter % 100 == 0:
            print("Test News Covered : ", counter)

    test_gwbowv_name = "TEST_SDV_" + str(num_clusters) + "cluster_" + str(
        num_features) + "feature_matrix_gmm_sparse.npy"

    print("Making sparse...")
    # Set the threshold percentage for making it sparse. 
    percentage = 0.04
    min_no = min_no * 1.0 / len(train["news"])
    max_no = max_no * 1.0 / len(train["news"])
    print("Average min: ", min_no)
    print("Average max: ", max_no)
    thres = (abs(max_no) + abs(min_no)) / 2
    thres = thres * percentage

    np.save(gwbowv_name, gwbowv)
    np.save(test_gwbowv_name, gwbowv_test)

    endtime = time.time() - start
    print("SDV created and dumped: ", endtime, "seconds.")
    print("Fitting a SVM classifier on labeled training data...")
    
    
    param_grid = [
        {'C': np.arange(5.25,5.5,0.25)}]
    scores = ['accuracy', 'recall_micro', 'f1_micro', 'precision_micro', 'recall_macro', 'f1_macro', 'precision_macro',
              'recall_weighted', 'f1_weighted', 'precision_weighted']  # , 'accuracy', 'recall', 'f1']
    for score in scores:
        strt = time.time()
        print("# Tuning hyper-parameters for", score, "\n")
        clf = GridSearchCV(LinearSVC(C=1), param_grid, cv=2, scoring='%s' % score,verbose=3)
        clf.fit(gwbowv, train["class"])
        print("Best parameters set found on development set:\n")
        print(clf.best_params_)
        print("Best value for ", score, ":\n")
        file1 = open("report.txt","w+")
        file1.write("Best value for ", score, ":\n"+ str(clf.best_score_)+"para"+str(clf.best_params_))
        print(clf.best_score_)
        Y_true, Y_pred = test["class"], clf.predict(gwbowv_test)
        print("Report")
        print(classification_report(Y_true, Y_pred, digits=6))
        print("Accuracy: ", clf.score(gwbowv_test, test["class"]))
        file1.write("TEST ACCURACY"+str(clf.score(gwbowv_test, test["class"])))
        print("Time taken:", time.time() - strt, "\n")
    endtime = time.time()
    print("Total time taken: ", endtime - start, "seconds.")

    print("********************************************************")
