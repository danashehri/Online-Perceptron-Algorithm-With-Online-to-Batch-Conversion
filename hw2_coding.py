import pandas as pd
import math
import numpy as np
import nltk
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from collections import Counter
from scipy.sparse import dok_matrix
from collections import defaultdict
import random
from scipy.sparse import hstack
from itertools import chain
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 1:
    print("must write command line: python3 hw2_coding.py")
    quit() 

df_te = pd.read_csv('reviews_te.csv')
df_tr =  pd.read_csv('reviews_tr.csv')

#nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

############################################################################################################
def perceptron(array, items, training_documents):
    w = np.array([0 for w in items])
    np.append(w, 0)
    all_weights = dok_matrix((2*len(training_documents), len(items)))
    indices = np.arange(array.shape[0])
    random.shuffle(indices)
    shuffled_matrix = array[list(indices)]
    round = 1
    while round <= 2:
        for d in range(len(training_documents)):
          x = shuffled_matrix[d,:-1].toarray() 
          np.append(x, 1)
          y = 1 if shuffled_matrix[d,-1] == 1 else -1
          fx = -1 if np.sign(np.dot(w, x.transpose())) < 0 else 1 
          if fx != y:
            w = w + y*x
            all_weights[d,:] = csr_matrix(w)          
          #print(d)
        #print("--------- round #", round, " done")  
        round += 1
        if round != 2:
          random.shuffle(indices)
          shuffled_matrix = shuffled_matrix[list(indices)]
    return all_weights
############################################################################################################
def getAvg(all_weights, items, training_documents):
  sum = np.expand_dims(np.array([float(0) for w in items]), axis=0)
  for w in range(all_weights.shape[0]):
    sum += all_weights[w,:].toarray()
  avg_weight = sum/len(training_documents)
  return avg_weight
############################################################################################################
def accuracy_score(items, data, avg_weight, labels):
  error = 0
  correct = 0
  for d, doc in enumerate(data):
    cleaned_doc = Counter(doc.split())
    x = np.array([cleaned_doc[word] if word in cleaned_doc else 0 for word in items.keys()])
    x = np.expand_dims(x, axis=0)
    y = 1 if labels[d] == 1 else -1  
    fx = -1 if np.sign(np.dot(avg_weight, x.transpose())) < 0 else 1
    if fx != y:
      error += 1
    else:
      correct += 1
  accuracy = correct / len(data) * 100
  return accuracy

def bgs_accuracy_score(items, data, avg_weight, labels):
  error = 0
  correct = 0
  for d, doc in enumerate(data):
    doc = doc.split()
    cleaned_doc = Counter(zip(doc[:-1], doc[1:]))
    x = np.array([cleaned_doc[bigram] if bigram in cleaned_doc else 0 for bigram in items.keys()])
    x = np.expand_dims(x, axis=0)
    y = 1 if labels[d] == 1 else -1  
    fx = -1 if np.sign(np.dot(avg_weight, x.transpose())) < 0 else 1
    if fx != y:
      error += 1
    else:
      correct += 1
  accuracy = correct / len(data) * 100
  return accuracy

training_documents = df_tr["text"].to_numpy()#[:10000]
test_documents = df_te["text"].to_numpy()#[:10000]
tr_labels = df_tr["label"].to_numpy()#[:10000]
te_labels = df_te["label"].to_numpy()#[:10000]
data = np.concatenate((training_documents, test_documents), axis=0)
labels = np.concatenate((tr_labels, te_labels), axis=0)
data_splits = []
training, test, tr_labels, te_labels = train_test_split(data, labels, train_size=0.60,test_size=0.40, random_state=101)
data_splits.append((training, test, tr_labels, te_labels))
training, test, tr_labels, te_labels = train_test_split(data, labels, train_size=0.70,test_size=0.30, random_state=101)
data_splits.append((training, test, tr_labels, te_labels))
training, test, tr_labels, te_labels = train_test_split(data, labels, train_size=0.75,test_size=0.25, random_state=101)
data_splits.append((training, test, tr_labels, te_labels))
training, test, tr_labels, te_labels = train_test_split(data, labels, train_size=0.80,test_size=0.20, random_state=101)
data_splits.append((training, test, tr_labels, te_labels))

total_accuracy_scores_with_unigrams = []
total_accuracy_scores_with_tfidf = []
total_accuracy_scores_with_bigrams = []

counter = 1
ps = PorterStemmer()
for sp in data_splits:   
    training, test, tr_labels, te_labels = sp[0], sp[1], sp[2], sp[3]
    all_data = np.concatenate((training, test), axis=0)
    #all_data = list(map(str.split, all_data))
    #all_data = list(map(str.split, list(all_data)))
    all_labels = np.concatenate((tr_labels, te_labels), axis=0)
    print("------------- starting data split #", counter, " -------------")
    print("------------- starting unigram data representation ------------- ")
   
    #Counting word frequencies over all documents
    words = Counter(chain([ps.stem(w) for d in all_data for w in d.split()]))
 
    #print("words = ", len(words))
    #print("avg = ", sum(words.values())/len(words))
    #removing un-important words
    all_words = Counter()
    for k,v in words.items():
      if k.isalpha() and len(k) > 3 and k not in stopwords:
        if v <= round(sum(words.values())/len(words)):
          all_words[k] = v

    #print("all_words = ", len(all_words))
    print("done with unigram vocabulary..")
    
    def unigram_rep(d, doc):
        vec = np.array(np.bincount([i for i,word in enumerate(all_words) if word in doc.split()], None, len(all_words)))
        uni_matrix[d,:] = csr_matrix(vec)
        #print(d)
        return uni_matrix

    #getting all unique words from dataset
    uni_matrix = dok_matrix((len(training), len(all_words)))
    d = [index for index in range(len(training))]
    unigram_vectors = list(map(unigram_rep, d, training))[0]

    labels_column = csr_matrix(tr_labels)
    an_array = hstack((unigram_vectors, labels_column.transpose()))
    uni_array = an_array.todok()
    print("finished unigram vectorization..")
    uni_weights = perceptron(uni_array, all_words, training)
    print("finished running perceptron..")
    uni_avg = getAvg(uni_weights, all_words, training)
    print("finished computing average weight..")
    uni_accuracy = accuracy_score(all_words, all_data, uni_avg, all_labels)
    print("accuracy score with unigrams = ", uni_accuracy)
    ##########################################################################################################
    ########################### tf-idf representation #######################################################
    print("------------- starting tf-idf data representation ------------- ")
    def tfidf_word_counts(doc):
        for word in doc.split(): 
            word_counts[word] += 1
        return word_counts 

    def tfidf_rep(d, doc):
        vec = np.array([math.log(len(training)/word_counts[word], 10) if word in doc.split() else 0 for word in all_words.keys()])
        tf_idf_matrix[d,:] = csr_matrix(vec)
        return tf_idf_matrix

    #counting specific word frequency in all training documents
    word_counts = defaultdict(int)
    word_counts = list(map(tfidf_word_counts, training))[0]

    #tfidf representation
    tf_idf_matrix = dok_matrix((len(training), len(all_words)))
    d = [index for index in range(len(training))]
    tf_idf_vectors = list(map(tfidf_rep, d, training))[0]
    an_array = hstack((tf_idf_vectors, labels_column.transpose()))
    tfidf_array = an_array.todok()
    print("finished tfidf vectorization..")
    tfidf_weights = perceptron(tfidf_array, all_words, training)
    print("finished running perceptron..")
    tfdidf_avg = getAvg(tfidf_weights, all_words, training)
    print("finished computing average weight..")
    tfidf_accuracy = accuracy_score(all_words, all_data, tfdidf_avg, all_labels)
    print("accuracy score with tfidf = ", tfidf_accuracy)
    ##########################################################################################################
    ########################### bigram representation ######################################################
    print("------------- starting bigram data representation ------------- ")
    def unique_bigrams(doc):
        # cleaned_doc = []
        # for word in doc.split(): 
        #     if word.isalpha() and len(word) > 3 and word not in set(stopwords.words('english')):
        #       cleaned_doc.append(word)
        doc = doc.split()
        for bg in zip(doc[:-1], doc[1:]):
          bigrams[bg] += 1
        return bigrams

    def bigram_rep(d, doc):
        doc = doc.split()
        vec = np.array(np.bincount([i for i,bg in enumerate(all_bigrams) if bg in zip(doc[:-1], doc[1:])], None, len(all_bigrams)))
        bigrams_matrix[d,:] = csr_matrix(vec)
        return bigrams_matrix

    bigrams = defaultdict(int)
    bigrams = list(map(unique_bigrams, all_data))[0]
    #print("bigrams = ", len(bigrams))
    #print("avg = ", sum(bigrams.values())/len(bigrams))
    all_bigrams = defaultdict(int)
    for k,v in bigrams.items():
      if v <= round(sum(bigrams.values())/len(bigrams)):
        all_bigrams[k] = v
    #print("all_bigrams =",len(all_bigrams))
    print("done with bigrams vocabulary..")
    bigrams_matrix = dok_matrix((len(training), len(all_bigrams)))
    d = [index for index in range(len(training))]
    bigrams_vectors = list(map(bigram_rep, d, training))[0]
    an_array = hstack((bigrams_vectors, labels_column.transpose()))
    bgs_array = an_array.todok()
    print("finished bigram vectorization..")
    bgs_weights = perceptron(bgs_array, all_bigrams, training)
    print("finished running perceptron..")
    bgs_avg = getAvg(bgs_weights, all_bigrams, training)
    print("finished computing the average weight..")
    bgs_accuracy = bgs_accuracy_score(all_bigrams, all_data, bgs_avg, all_labels)
    print("accuracy score with bigrams = ", bgs_accuracy)
    ##########################################################################################################
    total_accuracy_scores_with_unigrams.append(uni_accuracy)
    total_accuracy_scores_with_tfidf.append(tfidf_accuracy)
    total_accuracy_scores_with_bigrams.append(bgs_accuracy)
    counter += 1
    

print(total_accuracy_scores_with_unigrams)
print(total_accuracy_scores_with_tfidf)
print(total_accuracy_scores_with_bigrams)


splits = ["60/40", "70/30", "75/25", "80/20"]

plt.plot(splits, total_accuracy_scores_with_unigrams)
plt.plot(splits, total_accuracy_scores_with_tfidf)
plt.plot(splits, total_accuracy_scores_with_bigrams)
plt.xlabel("train-test-splits")
plt.ylabel("Linear Classifier Accuracy Score")
plt.legend(["Unigrams Representation", "TF-IDF Representation", "Bigrams Representation"])
plt.show()

##### Finding top 10 positive and negative words via Unigrams representation of data
print("Looking for the top-10 positive and negative words...")
training_documents = df_tr["text"].to_numpy()
test_documents = df_te["text"].to_numpy()
tr_labels = df_tr["label"].to_numpy()
te_labels = df_te["label"].to_numpy()
data = np.concatenate((training_documents, test_documents), axis=0)
labels = np.concatenate((tr_labels, te_labels), axis=0)
training, test, tr_labels, te_labels = train_test_split(data, labels, train_size=0.75,test_size=0.25, random_state=101)
all_data = np.concatenate((training, test), axis=0)
all_labels = np.concatenate((tr_labels, te_labels), axis=0)

def examples(training, tr_labels, all_data, all_labels):
    ps = PorterStemmer()
    words = Counter(chain([w for d in all_data for w in d.split()]))
    all_words = Counter()
    for k,v in words.items():
      if k.isalpha() and len(k) > 3 and k not in stopwords:
        if v <= round(sum(words.values())/len(words)):
          all_words[k] = v
    
    def unigram_rep(d, doc):
        vec = np.array(np.bincount([i for i,word in enumerate(all_words) if word in doc.split()], None, len(all_words)))
        uni_matrix[d,:] = csr_matrix(vec)
        return uni_matrix

    uni_matrix = dok_matrix((len(training), len(all_words)))
    d = [index for index in range(len(training))]
    unigram_vectors = list(map(unigram_rep, d, training))[0]
    labels_column = csr_matrix(tr_labels)
    an_array = hstack((unigram_vectors, labels_column.transpose()))
    uni_array = an_array.todok()
    uni_weights = perceptron(uni_array, all_words, training)
    uni_avg = getAvg(uni_weights, all_words, training)#[0:(len(uni_avg)-1)]


    neg_words = []
    pos_words = []
    wn = uni_avg[0].copy()
    wp = uni_avg[0].copy()
    vocab = list(all_words)
    vocab_neg = list(all_words)
    vocab_pos = list(all_words)
  
    for i in range(10):
      idx_min = np.argmin(wn)
      idx_max = np.argmax(wp) 
      n = vocab[idx_min]
      p = vocab[idx_max]
      neg_words.append(n)
      pos_words.append(p)
      wn = np.delete(wn, idx_min)
      wp = np.delete(wp, idx_max)
      vocab_neg = np.delete(vocab_neg, idx_min)
      vocab_pos = np.delete(vocab_pos, idx_max)

    return neg_words, pos_words
neg_words, pos_words = examples(training, tr_labels, all_data, all_labels)
print("Words with lowest weights: ", neg_words)
print("Words with highest weights: ", pos_words)