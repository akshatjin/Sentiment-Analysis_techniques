import nltk
import glob
import os
import numpy as np
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from sklearn.metrics import accuracy_score
from collections import Counter
from collections import defaultdict



#Creating the design matrix X and y
X_train_text = []
Y_train = []
X_test_text =[]
Y_test =[]
Vocab = {}
VocabFile = "aclImdb/imdb.vocab"
Word2vecFile = 'word2vec.txt'
gloveFile = 'glove.txt'
def CreateVocab():
	with open(VocabFile) as f:
		words = f.read().splitlines()
		#print(len(words))
		stop_words = set(stopwords.words('english'))
		i=0
		for word in words:
			if word not in stop_words:
				Vocab[word] = i
				i+=1
		#print(len(Vocab))



#Manual cleaning
def clean_review(text):
# split into tokens: 
	tokens = word_tokenize(text) #list of words
# 	print(tokens[:100])
# convert to lower case
	tokens = [w.lower() for w in tokens]
# remove punctuation from each word
	table = str.maketrans('', '', string.punctuation)
	stripped = [w.translate(table) for w in tokens]
# remove remaining tokens that are not alphabetic
	words = [word for word in stripped if word.isalpha()]
#remove stop-words
	stop_words = set(stopwords.words('english'))
	words = [w for w in words if not w in stop_words]
	return words


def BoWMatrix(docs):
	'''docs should be a list of strings'''
	vectorizer = CountVectorizer(binary=True,vocabulary = Vocab)
	Doc_Term_matrix = vectorizer.fit_transform(docs)
	return Doc_Term_matrix

def TfidfMatrix(docs):
	'''docs should be a list of strings'''
	vectorizer = TfidfVectorizer(vocabulary = Vocab,norm = 'l1')
	Doc_Term_matrix = vectorizer.fit_transform(docs)
	return Doc_Term_matrix


def NB(X,Y_train,Xtest,Y_test,met):
	if met == "Bow":
		clf = BernoulliNB()
	elif met == "Tfidf":
		clf = MultinomialNB()
	else:
		clf = GaussianNB()
	clf.fit(X,Y_train)
	pred = clf.predict(Xtest)
	acc = accuracy_score(Y_test,pred)
	print("NaiveBayes + " + met + " : " + str(acc*100) + "%")

def LR(X,Y_train,Xtest,Y_test,met):
	lr = LogisticRegression()
	lr.fit(X,Y_train)
	pred = lr.predict(Xtest)
	acc = accuracy_score(Y_test,pred)
	print("LogisticRegression + " + met + " : " + str(acc*100) + "%")

def SVM(X,Y_train,Xtest,Y_test,met):
	clf = LinearSVC()
	clf.fit(X,Y_train)

	# print("Saving SVM model to file...........")
	# pickle_out = open("SVM" + met + ".pickle","wb")
	# pickle.dump(clf, pickle_out)
	# pickle_out.close()

	# print("Loading SVM model...")
	# pickle_in = open("SVM" + met + ".pickle","rb")
	# clf = pickle.load(pickle_in)

	pred = clf.predict(Xtest)
	acc = accuracy_score(Y_test,pred)
	print("SVM + " + met + " : " + str(acc*100) + "%")

def NN(X,Y_train,Xtest,Y_test,met):
	nn = MLPClassifier(hidden_layer_sizes=(100,100),activation='relu',max_iter=250)
	nn.fit(X,Y_train)
	print("Saving NN model to file....")
	pickle_out = open("NeuralNet" + met + ".pickle","wb")
	pickle.dump(nn, pickle_out)
	pickle_out.close()

	print("Loading NN model...")
	pickle_in = open("NeuralNet" + met + ".pickle","rb")
	nn = pickle.load(pickle_in)

	pred = nn.predict(Xtest)
	acc = accuracy_score(Y_test,pred)
	print("FFN + " + met + " : " + str(acc*100) + "%")

def Getbowvec(X_train_text,Y_train,X_test_text,Y_test):
	#Bag Of Words
	X = BoWMatrix(X_train_text)
	Xtest = BoWMatrix(X_test_text)
	return X,Xtest

def Gettfidfvec(X_train_text,Y_train,X_test_text,Y_test):
	#Tfidf
	X = TfidfMatrix(X_train_text)
	Xtest = TfidfMatrix(X_test_text)
	return X,Xtest

def Word_to_vec(X_train_text,Y_train,X_test_text,Y_test):
	#model = Word2Vec.load('mywordvecs.bin')
	#checking correctness
	print("Loading Word2Vec Model....")
	f = open(Word2vecFile,'r')
	Dim = 0
	model = {}
	for line in f:
		splitLine = line.split(' ')
		Dim = len(splitLine) -1
		#print(Dim)
		word = splitLine[0]
		#print(word)
		embedding = np.array([float(val) for val in splitLine[1:]])
		model[word] = embedding
	print('Word vectors loaded')
	print("Embedding being done,this might take a while :")
	X=[]
	Xtest = []
	X1 =[]
	Xtest1 =[]
	new_corpus = []
	NumDocs = len(X_train_text)
	word_idf = defaultdict(lambda: 0)
	for sen in X_train_text:
		tokens = clean_review(sen)
		#print(type(tokens))
		words = set(tokens)
		for word in words:
			word_idf[word] += 1
			word_idf[word] = np.log(NumDocs*1.0 / 1.0*(1.0 + word_idf[word]))
		new_corpus.append(tokens)
	#print(model[new_corpus[0][0]])
	#print(len(model[new_corpus[0][0]]))


	
	for sen in new_corpus:
		dummy = np.zeros(Dim)
		dummy1 = np.zeros(Dim)
		temp = Counter(sen)
		for word in sen:
			if word in model.keys():
				dummy += np.array(model[word])
				dummy1 += temp[word]*word_idf[word]*np.array(model[word])
		dummy /= len(sen)
		dummy = list(dummy)
		dummy1 /= len(sen)
		dummy1 = list(dummy1)
		X.append(dummy)
		X1.append(dummy1)



	test_corpus = []
	for sen in X_test_text:
		tokens = clean_review(sen)

		test_corpus.append(tokens)
	#print(model[test_corpus[0][0]])
	#print(len(model[test_corpus[0][0]]))
	
	for sen in test_corpus:
		dummy = np.zeros(Dim)
		dummy1 = np.zeros(Dim)
		temp = Counter(sen)
		for word in sen:
			if word in model.keys():
				dummy += np.array(model[word])
				dummy1 += temp[word]*word_idf[word]*np.array(model[word])
		dummy /= len(sen)
		dummy = list(dummy)
		dummy1 /= len(sen)
		dummy1 = list(dummy1)
		Xtest.append(dummy)
		Xtest1.append(dummy1)
	#print(len(X))
	return X,Xtest,X1,Xtest1


def GloVe(X_train_text,Y_train,X_test_text,Y_test):
	#pickle_in = open("dict.pickle","rb")
	#model = pickle.load(pickle_in)
	print("Loading Glove Model......")
	f = open(gloveFile,'r')
	Dim = 0
	model = {}
	for line in f:
		splitLine = line.split(' ')
		Dim = len(splitLine)-1
		#print(Dim)
		word = splitLine[0]
		#print(word)
		embedding = np.array([float(val) for val in splitLine[1:]])
		model[word] = embedding
    
	print('Glove vectors loaded')
	print("Embedding being done,this might take a while :")
	X=[]
	Xtest = []
	X1 =[]
	Xtest1 =[]
	new_corpus = []
	word_idf = defaultdict(lambda: 0)
	NumDocs = len(X_train_text)
	for sen in X_train_text:
		tokens = clean_review(sen)
		words = set(tokens)
		for word in words:
			word_idf[word] += 1
			word_idf[word] = np.log(NumDocs*1.0/ 1.0*(1 + word_idf[word]))
		new_corpus.append(tokens)
	# print(model[new_corpus[0][0]])
	# print(len(model[new_corpus[0][0]]))


	
	for sen in new_corpus:
		dummy = np.zeros(Dim)
		dummy1 = np.zeros(Dim)
		temp = Counter(sen)
		for word in sen:
			if word in model.keys():
				#print(np.array(model[word]).ndim)
				dummy += np.array(model[word])
				dummy1 += temp[word]*word_idf[word]*np.array(model[word])
		dummy /= len(sen)
		dummy = list(dummy)
		dummy1 /= len(sen)
		dummy1 = list(dummy1)
		X.append(dummy)
		X1.append(dummy1)
	#print("Training corpus embedded")



	test_corpus = []
	for sen in X_test_text:
		tokens = clean_review(sen)
		test_corpus.append(tokens)
	# print(model[test_corpus[0][0]])
	# print(len(model[test_corpus[0][0]]))
	
	for sen in test_corpus:
		dummy = np.zeros(Dim)
		dummy1 = np.zeros(Dim)
		temp = Counter(sen)
		for word in sen:
			if word in model.keys():
				dummy += np.array(model[word])
				dummy1 += temp[word]*word_idf[word]*np.array(model[word])
		dummy /= len(sen)
		dummy = list(dummy)
		dummy1 /= len(sen)
		dummy1 = list(dummy1)
		Xtest.append(dummy)
		Xtest1.append(dummy1)
	#print("Test corpus embedded")
	#print(len(X))
	return X,Xtest,X1,Xtest1

def Doc2vec(X_train_text,Y_train,X_test_text,Y_test):
	model = Doc2Vec.load('doc2vec.model')
	X = []
	Xtest =[]
	for i,l in enumerate(X_train_text):
		temp = "review" + "_" + str(i)
		X.append(model.docvecs[temp])
	for i,l in enumerate(X_test_text):
		temp = "test" + "_" + str(i)
		Xtest.append(model.docvecs[temp])
	print("Doc2Vec built")
	return X,Xtest




##########################################################################################################
####----------------------------------------MAIN CODE-------------------------------------------------####
##########################################################################################################

#Open all training files:
path1 = 'aclImdb/train/pos/*.txt'
path2 = 'aclImdb/train/neg/*.txt'
path3 = 'aclImdb/test/pos/*.txt'
path4 = 'aclImdb/test/neg/*.txt'

files1 = glob.glob(path1)
files2 = glob.glob(path2)
files3 = glob.glob(path3)
files4 = glob.glob(path4)

print("Loading data...\n")
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
CreateVocab();
print("Vocab created")


print("Creating embeddings..................")
print(" ")
#Get BagofWords Matrix
print("Bag of Words is being built...")
X,Xtest = Getbowvec(X_train_text,Y_train,X_test_text,Y_test)
# scipy.sparse.save_npz("BowEmbeddingTr.npz",X)
# scipy.sparse.save_npz("BowEmbeddingTe.npz",Xtest)
#Get Tf-df Matrix
print("Tf-idf is being built...")
X1,Xtest1 = Gettfidfvec(X_train_text,Y_train,X_test_text,Y_test)
# scipy.sparse.save_npz("TfidfEmbeddingTr.npz",X1)
# scipy.sparse.save_npz("TfidfEmbeddingTe.npz",Xtest1)
#Word2vec
print("Word2vec is being built...")
X2,Xtest2,X3,Xtest3 = Word_to_vec(X_train_text,Y_train,X_test_text,Y_test)
#GloVe
print("Glove is being built...")
X4,Xtest4,X5,Xtest5  = GloVe(X_train_text,Y_train,X_test_text,Y_test)
#Doc2Vec
print("Doc2Vec is being built...")
X6,Xtest6 = Doc2vec(X_train_text,Y_train,X_test_text,Y_test)
#################################################################
#################	CALLING ALL MODELS!! ########################
#################################################################

#NB
print("Naive Bayes =>")
NB(X,Y_train,Xtest,Y_test,"Bow")
NB(X1,Y_train,Xtest1,Y_test,"Tfidf")
NB(X2,Y_train,Xtest2,Y_test,"Word2vec")
NB(X3,Y_train,Xtest3,Y_test,"Word2vec-tfidf weight")
NB(X4,Y_train,Xtest4,Y_test,"Glove")
NB(X5,Y_train,Xtest5,Y_test,"Glove-tfidf weight")
NB(X6,Y_train,Xtest6,Y_test,"Doc2Vec")

#LR
print("LogisticRegression =>")
LR(X,Y_train,Xtest,Y_test,"Bow")
LR(X1,Y_train,Xtest1,Y_test,"Tfidf")
LR(X2,Y_train,Xtest2,Y_test,"Word2vec")
LR(X3,Y_train,Xtest3,Y_test,"Word2vec-tfidf weight")
LR(X4,Y_train,Xtest4,Y_test,"Glove")
LR(X5,Y_train,Xtest5,Y_test,"Glove-tfidf weight")
LR(X6,Y_train,Xtest6,Y_test,"Doc2Vec")

#SVM
print("SVM =>")
SVM(X,Y_train,Xtest,Y_test,"Bow")
SVM(X1,Y_train,Xtest1,Y_test,"Tfidf")
SVM(X2,Y_train,Xtest2,Y_test,"Word2vec")
SVM(X3,Y_train,Xtest3,Y_test,"Word2vec-tfidf weight")
SVM(X4,Y_train,Xtest4,Y_test,"Glove")
SVM(X5,Y_train,Xtest5,Y_test,"Glove-tfidf weight")
SVM(X6,Y_train,Xtest6,Y_test,"Doc2Vec")

#Neural Network
print("FeedForward Network =>")
NN(X,Y_train,Xtest,Y_test,"Bow")
NN(X1,Y_train,Xtest1,Y_test,"Tfidf")
NN(X2,Y_train,Xtest2,Y_test,"Word2vec")
NN(X3,Y_train,Xtest3,Y_test,"Word2vec-tfidf weight")
NN(X4,Y_train,Xtest4,Y_test,"Glove")
NN(X5,Y_train,Xtest5,Y_test,"Glove-tfidf weight")
NN(X6,Y_train,Xtest6,Y_test,"Doc2Vec")

