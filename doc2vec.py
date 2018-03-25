import gensim
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import multiprocessing
from sklearn import utils
import os
import glob
import errno
import string
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


print("Loading data....")

X_train_text = []
Y_train = []
X_test_text =[]
Y_test =[]

#Open all training files:
path1 = 'aclImdb/train/pos/*.txt'
path2 = 'aclImdb/train/neg/*.txt'
path3 = 'aclImdb/test/pos/*.txt'
path4 = 'aclImdb/test/neg/*.txt'

files1 = glob.glob(path1)
files2 = glob.glob(path2)
files3 = glob.glob(path3)
files4 = glob.glob(path4)

#Manual cleaning
def clean_review(data):
# split into tokens: 
    D = []
    for text in data:
      tokens = word_tokenize(text) #list of words
#   print(tokens[:100])
# convert to lower case
      tokens = [w.lower() for w in tokens]
# remove punctuation from each word
      table = str.maketrans('', '', string.punctuation)
      stripped = [w.translate(table) for w in tokens]
# remove remaining tokens that are not alphabetic
      words = [word for word in stripped if word.isalpha()]
      D.append(words)
    return D

#Positive labels
for i,filename in enumerate(files1):
	#print(i)
	f = open(filename,"r+")
	text = f.read()
	f.close()
	#text = clean_review(text)
	X_train_text.append(text) #list of strings
	Y_train.append(1)

#Neg labels
for j,filename in enumerate(files2):
	#print(j)
	f = open(filename,"r+")
	text = f.read()
	f.close()
	#text = clean_review(text)
	X_train_text.append(text) #listof strings
	Y_train.append(0)

#Test labels +
for k,filename in enumerate(files3):
	#print(j)
	f = open(filename,"r+")
	text = f.read()
	f.close()
	X_test_text.append(text) #listof strings
	Y_test.append(1)

#Test labels +
for l,filename in enumerate(files4):
	#print(j)
	f = open(filename,"r+")
	text = f.read()
	f.close()
	X_test_text.append(text) #listof strings
	Y_test.append(0)

print("Done loading data")

print(len(X_train_text))

#This function does all cleaning of data using two objects above
print("Cleaning data....")
X_train_text = clean_review(X_train_text)
print("After Cleaning:",len(X_train_text))
X_test_text = clean_review(X_test_text)


def LabelRev(reviews,label_string):
    result = []
    prefix = label_string
    for i, t in enumerate(reviews):
    	# print(t)
    	result.append(LabeledSentence(t, [prefix + '_%s' % i]))
    return result

LabelledXtrain = LabelRev(X_train_text,"review")    
LabelledXtest = LabelRev(X_test_text,"test")

LabelledData = LabelledXtrain + LabelledXtest

modeld2v = Doc2Vec(dm=0, min_count=2, alpha=0.065, min_alpha=0.065)
modeld2v.build_vocab([x for x in tqdm(LabelledData)])

print("Training the Doc2Vec Model.....")
for epoch in range(50):
	print("epoch : ",epoch)
	modeld2v.train(utils.shuffle([x for x in tqdm(LabelledData)]), total_examples=len(LabelledData), epochs=1)
	modeld2v.alpha -= 0.002
	modeld2v.min_alpha = modeld2v.alpha


print("Saving Doc2Vec Model....")
modeld2v.save('doc2vec.model')    

print(len(review_set))
print(len(labeled_train_reviews))
