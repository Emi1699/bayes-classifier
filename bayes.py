import csv
# import re
from copy import deepcopy
from collections import Counter
import numpy as np

#get list of stopwords
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
stopWordsList = stopwords.words('english')

class NaiveBayes:
	def __init__(self):
		self.preprocessing = Preprocessing()

		print("reading data from tsv file..\n")
		self.training_data = self.preprocessing.readFile("train.tsv")
		self.dev_data = self.preprocessing.readFile("dev.tsv")
		self.test_data = self.preprocessing.readFile("test.tsv")

		print("mapping 5-value data to 3-value data\n")
		data_5values, data_3values = self.preprocessing.map_5to3(self.training_data)

		self.priors_3values = self.priors(data_3values, 3)
		self.priors_5values = self.priors(data_5values, 5)

		print(self.priors_3values)
		print(self.priors_5values)

		#split sentences into words (features), stored in dictionaries (this makes computing likelihoods easier)
		#then filter out the stopwords
		self.features_3values = self.preprocessing.tokenizeSentences(data_3values, 3)
		self.features_5values = self.preprocessing.tokenizeSentences(data_5values, 5)

		#count number of distinct features (|V|) used for the Laplace smoothing technique
		self.distinctFeatures = self.distinctFeatures(self.features_5values)
		self.numberDistinctFeatures = len(self.distinctFeatures)

		#apply stopword removal
		self.features_3valuesSW = self.preprocessing.applyStopwords(self.features_3values)
		self.features_5valuesSW = self.preprocessing.applyStopwords(self.features_5values)

		self.likelihoods_3values = self.trainingLikelihoods(self.features_3values, 3)
		self.likelihoods_5values = self.trainingLikelihoods(self.features_5values, 5)

		feats = ["hard", "good", "aggressive"]

		# print(self.likelihood("hard good aggressive", 5))
		# print("\n\n")

		# for feat in feats:
		# 	print(feat + "\n")
		# 	for i in range(0, 5):
		# 		if feat in self.likelihoods_5values[str(i)]:
		# 			print(str(i) + " ==> " + str(self.likelihoods_5values[str(i)][feat]))

		print(self.computeAccuracy(5))


	#compute posterios probabilities for each class
	def posterior(self, text, values):

		if values == 3:
			posteriors = [0, 0, 0]
			likelihoods = self.likelihood(text, 3)

			for value in range(0, 3):
				likelihoodsProduct = 1

				for feature in likelihoods[str(value)]:
					likelihoodsProduct *= likelihoods[str(value)][feature]

				posteriors[value] = self.priors_3values[value] * likelihoodsProduct

		else:
			posteriors = [0, 0, 0, 0, 0]
			likelihoods = self.likelihood(text, 5)

			for value in range(0, 5):
				likelihoodsProduct = 1

				for feature in likelihoods[str(value)]:
					likelihoodsProduct *= likelihoods[str(value)][feature]

				posteriors[value] = self.priors_5values[value] * likelihoodsProduct

		return argmax(posteriors)

	#compute prior probabilities for each class
	def priors(self, data, values):
		#!!! data before tokenizing sentences
		datapoints = len(data)
		# print(data)

		if values == 3:
			# print("computing 3-valued data priors..")
			priors = [0, 0, 0]
		else:
			# print("computing 5-valued data priors..")
			priors = [0, 0, 0, 0, 0]

		#!!! data before tokenizing sentences
		for line in data:
			priors[int(line[2])] += 1

		priors[:] = [x/datapoints for x in priors]

		return priors

	#compoute likelihood for each feature in the training set (!using Laplace smoothing!)
	def trainingLikelihoods(self, data, values):
		if values == 3:
			# print("computing 3-valued data likelihoods..")
			likelihoods = dict()
			for i in range(0, 3):
				likelihoods.setdefault(str(i), {})
		else:
			# print("computing 5-valued data likelihoods..")
			likelihoods = dict()
			for i in range(0, 5):
				likelihoods.setdefault(str(i), {})

		#count number of distinct features (|V|) used for the Laplace smoothing technique

		#!!! data after tokenizing sentences
		for value in data:
			# print(value)
			totalFeatures = len(data[value]) #total features for a given class
			# print(totalFeatures)
			featureFrequency = Counter(data[value])
			# print(featureFrequency)

			#compute likelihoods and store them in a dictionary
			#each key is a class (sentiment)
			#for each class (S), there will be a dictionary which will hold features (T) and their respective likelihood -> p(T/S)
			likelihoods[value] = {feature:(featureFrequency[feature] + 1)/(totalFeatures + self.numberDistinctFeatures) for feature in featureFrequency}

		# print(likelihoods)
		return likelihoods

	#compute likelihood for text to be classified
	def likelihood(self, text, values):
		features =  self.preprocessing.tokenizeSentence(text)
		distinctFts = self.numberDistinctFeatures
		likelihoods = dict()

		#if a text to be classified contains features that were not present in the training set
		#then we need to recompute |V|
		for feature in features:
			if feature not in self.distinctFeatures:
				distinctFts += 1

		if values == 5: #5-class data
			for i in range(0, 5):
				likelihoods.setdefault(str(i), {})

			for feature in features:
				for value in self.likelihoods_5values:
					if feature in self.likelihoods_5values[value]:
						likelihoods[value].update({feature:self.likelihoods_5values[value][feature]})
					else:
						#the feature has not been seen before (i.e. not present in the training data)
						likelihoods[value].update({feature:(1/(len(self.features_5values) + distinctFts))})

		else: #3-class data
			for i in range(0, 3):
				likelihoods.setdefault(str(i), {})

			for feature in features:
				for value in self.likelihoods_3values:
					if feature in self.likelihoods_3values[value]:
						likelihoods[value] = {feature:self.likelihoods_3values[value][feature]}
					else:
						#the feature has not been seen before (i.e. not present in the training data)
						likelihoods[value].update({feature:(1/(len(self.features_3values) + distinctFts))})

		return likelihoods

	#count the number of distinct features in the training set
	def distinctFeatures(self, data):
		return set(feature for value in data for feature in data[value])

	def computeAccuracy(self, values):
		devDataSentiments = dict()
		posteriors = dict()
		total = len(self.dev_data)

		for i in range(0, len(self.dev_data)):
			devDataSentiments.update({self.dev_data[i][0]:self.dev_data[i][2]})
			posteriors.update({self.dev_data[i][0]:self.posterior(self.dev_data[i][1], values)})

		print(len(devDataSentiments))
		print(len(posteriors))

		rightSentiments = 0
		for sentenceID in devDataSentiments:
			if devDataSentiments[sentenceID] == posteriors[sentenceID]:
				rightSentiments += 1

		print(rightSentiments)

		accuracy = rightSentiments / total

		return accuracy

	
#class that deals with preprocessing steps (e.g. stopword removal)
class Preprocessing:
	def __init__(self):
		print("preprocessing\n")

	def map_5to3(self, data):
		data_5values = deepcopy(data)
		data_3values = []


		
		for line in data:
			if line[len(line) - 1] == '0' or line[len(line) -  1] == '1':
				line[len(line) - 1] = '0'
			elif line[len(line) -  1] == '2':
				line[len(line) - 1] = '1'
			else:
				line[len(line) - 1] = '2'

			data_3values.append(line)

		return (data_5values, data_3values)

	#tokenize each sentence in the training set
	#the reason for using dictionaries for storing these features is that 
	#it makes it easier to compute the required probabilities
	def tokenizeSentences(self, data, values):
		features = dict()

		if values == 5:
			print('\n#--> tokenizing 5-valued data <--#\n')
			for i in range(0, 5):
				features.setdefault(str(i),[])
		else:
			print('\n#--> tokenizing 3-valued data <--#\n')
			for i in range(0, 3):
				features.setdefault(str(i),[])

		for line in data:
			value = line[2] #here value stands for 'class' (i.e. the sentiment of the text)
			tokens = nltk.word_tokenize(line[1])
			features[value].extend(tokens)
			
		return features

	def tokenizeSentence(self, sentence):
		return nltk.word_tokenize(sentence)

	#remove stopwords from a dictionary of features
	def applyStopwords(self, features):
		print("applying stopwords..\n")
		for value in features:
			features[value] = [feature for feature in features[value] if feature.lower() not in stopWordsList]
		
		return features

	#extract data from tsv file
	def readFile(self, file):
		data = csv.reader(open(file), delimiter="\t")

		#put the data in a list in order to be processed accordingly
		training_data = []
		for line in data:
			training_data.append(line)

		return training_data[1:] #ignore the first line of the file

def argmax(arr):
	return max(range(len(arr)), key=lambda i: arr[i])

bayes = NaiveBayes()











