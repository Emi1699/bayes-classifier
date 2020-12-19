import csv

from copy import deepcopy
from collections import Counter
#get list of stopwords
import nltk

#this was used to create the stopWords.txt file
	# nltk.download('stopwords')
	# nltk.download('punkt')
	# from nltk.corpus import stopwords
	# stopWordsList = stopwords.words('english')

with open("stopWords.txt") as f:
    content = f.readlines()

stopWordsList = [x.strip() for x in content] 

class NaiveBayes:
	def __init__(self):
		self.preprocessing = Preprocessing()

		print("reading data from tsv file..\n")
		self.training_data = self.preprocessing.readFile("train.tsv")
		self.dev_data = self.preprocessing.readFile("dev.tsv")
		self.test_data = self.preprocessing.readFile("test.tsv")

		#map 5valued data to 3valued data
		self.data_train_5values, self.data_train_3values = self.preprocessing.map_5to3(self.training_data)
		self.data_dev_5values, self.data_dev_3values = self.preprocessing.map_5to3(self.dev_data)

		self.priors_3values = self.priors(self.data_train_3values, 3)
		self.priors_5values = self.priors(self.data_train_5values, 5)

		#split sentences into words (features), stored in dictionaries (this makes computing likelihoods easier)
		#then filter out the stopwords
		self.features_3values = self.preprocessing.tokenizeSentences(self.data_train_3values, 3)
		self.features_5values = self.preprocessing.tokenizeSentences(self.data_train_5values, 5)

		#apply stopword removal !!!!
		self.preprocessing.applyStopwords(self.features_3values)
		self.preprocessing.applyStopwords(self.features_5values)

		#count number of distinct features (|V|) used for the Laplace smoothing technique
		self.distinctFeatures = self.distinctFeatures(self.features_3values)
		self.numberDistinctFeatures = len(self.distinctFeatures)

		print("computing likelihoods....")
		self.likelihoods_3values = self.trainingLikelihoods(self.features_3values, 3)
		self.likelihoods_5values = self.trainingLikelihoods(self.features_5values, 5)

		# print("ALL WORDS model")
		# print(self.computeAccuracy('test', 3))
		# print(self.computeAccuracy('test', 5))
		# print(self.computeAccuracy('dev', 3))
		# print(self.computeAccuracy('dev', 5))

		# print("NO STOP-WORDS model")
		# print(self.computeAccuracy('test', 3))
		# print(self.computeAccuracy('test', 5))
		# print(self.computeAccuracy('dev', 3))
		# print(self.computeAccuracy('dev', 5))

		print("NO STOP-WORDS model")
		print(self.computeAccuracy('test', 3))
		print(self.computeAccuracy('test', 5))
		print(self.computeAccuracy('dev', 3))
		print(self.computeAccuracy('dev', 5))

		# print(self.likelihood("thanks good 60", 3))


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
		#the data in discussion is before tokenizing the sentences, as the data after tokenization doesn't
		#hold info about the total number of each instance of class (sentiment)
		datapoints = len(data)

		if values == 3:
			# print("computing 3-valued data priors..")
			priors = [0, 0, 0]
		else:
			# print("computing 5-valued data priors..")
			priors = [0, 0, 0, 0, 0]

		#!!! data before tokenizing sentences
		for line in data:
			priors[int(line[2])] += 1

		#divide each sum by the total number of sentences to get the priors
		priors[:] = [x/datapoints for x in priors]

		return priors

	#compoute likelihood for each feature in the training set (!using Laplace smoothing!)
	def trainingLikelihoods(self, data, values):
		if values == 3: #3-class  data
			likelihoods = dict()
			for i in range(0, 3):
				likelihoods.setdefault(str(i), {})
		else: #5-class data
			likelihoods = dict()
			for i in range(0, 5):
				likelihoods.setdefault(str(i), {})

		#data after tokenizing sentences
		for sentiment in data:
			#hold the number of times each feature appears in the corresponding sentiment set
			featureFrequency = Counter(data[sentiment])

			#for word in the training set
			for feature in self.distinctFeatures:
				if feature in featureFrequency: #the word is in the training set
					likelihood = (1 + featureFrequency[feature]) / (self.numberDistinctFeatures + len(data[sentiment]))
				else: #the word is not in the training set
					likelihood = 1 / (self.numberDistinctFeatures + len(data[sentiment]))

				likelihoods[sentiment].update({feature:likelihood})
				
		return likelihoods

	#compute likelihood for text to be classified (using Laplace smoothing)
	def likelihood(self, text, values):
		features =  self.preprocessing.tokenizeSentence(text)
		# distinctFts = self.numberDistinctFeatures
		likelihoods = dict()

		#if a text to be classified contains features that were not present in the training set
		#then we need to recompute |V|
		for feature in features:
			if feature not in self.distinctFeatures:
				self.numberDistinctFeatures += 1

		# print("dist feats afet qwerqerq")
		# print(self.numberDistinctFeatures)

		if values == 5: #5-class data
			for i in range(0, 5):
				likelihoods.setdefault(str(i), {})

			for sentiment in likelihoods:
				featureFrequency = Counter(self.features_5values[sentiment])

				for word in features:
					if word in self.distinctFeatures:
						likelihood = self.likelihoods_5values[sentiment][word]
					else:
						likelihood = 1 / (len(self.features_5values[sentiment]) + self.numberDistinctFeatures)

					likelihoods[sentiment].update({word:likelihood})

		else: #3-class data
			for i in range(0, 3):
				likelihoods.setdefault(str(i), {})

			for sentiment in likelihoods:
				featureFrequency = Counter(self.features_3values[sentiment])
				# print("feat freq for " + str(sentiment))
				# print(featureFrequency)

				for word in features:
					if word in self.features_3values[sentiment]:
						likelihood = self.likelihoods_3values[sentiment][word]
					else:
						likelihood = 1 / (len(self.features_3values[sentiment]) + self.numberDistinctFeatures)

					likelihoods[sentiment].update({word: likelihood})

		return likelihoods

	#return a list of all the distinct features in the training set
	def distinctFeatures(self, data):
		return set(feature for value in data for feature in data[value])

	def computeAccuracy(self, data, values):
		#this is where we store the dev file data, in order to extract the sentiments
		#we need a separate because the data changes based on what user wants to compute
		fileData = [] 

		#name of file where we will output the -sentenceId/prediction- pairs
		#to be set depending on what the user intends to compute
		outputFile = "" 

		#this is where we store the sentenceIDs and sentiments from the dev/test file
		#(I am aware there are no sentiment scores in the test files, I'm just using the same structure for symmetry reasons) 
		idSentiment_file = dict()

		#this is where we will store the predicted sentiments
		predictions = dict()

		#number of total reviews to be classified (to be set depending on whether we're using the dev or test set)
		total = None

		if data == 'dev':
			if values == 3:
				print('dev 3...')
				fileData = self.data_dev_3values
				outputFile = "dev_predictions_3classes_Emanuel_BULIGA.tsv"
				total = len(fileData)
			else:
				print('dev 5...')
				fileData = self.data_dev_5values
				outputFile = "dev_predictions_5classes_Emanuel_BULIGA.tsv"
				total = len(fileData)
		else:
			if values == 3:
				print('test 3...')
				fileData = self.test_data
				outputFile = "test_predictions_3classes_Emanuel_BULIGA.tsv"
				total = len(fileData)
			else:
				print('test 5...')
				fileData = self.test_data
				outputFile = "test_predictions_5classes_Emanuel_BULIGA.tsv"
				total  = len(fileData)

		for i in range(total):
			if fileData is self.test_data: #no sentiment value in test file. Just initialize all to 0 and ignore
				idSentiment_file.update({fileData[i][0]:0})
			else:
				idSentiment_file.update({fileData[i][0]:fileData[i][2]})
			predictions.update({fileData[i][0]:self.posterior(fileData[i][1], values)})

		rightSentiments = 0

		for sentenceID in predictions:
			appendToFile(outputFile, str(sentenceID) + "\t" + str(predictions[sentenceID]) + "\n")
			if fileData is not self.test_data:
				if str(idSentiment_file[sentenceID]) == str(predictions[sentenceID]):
					rightSentiments += 1

		accuracy = rightSentiments * 100 / total

		return str(accuracy) + "%"

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

		return training_data[1:] #ignore the first line of the file, as it only specifies how the data is structured

	#difference of 2 sets
	def listDiff(l1, l2):
		return [x for x in l1 if x not in l2]

#find the position of the biggest value in a list
def argmax(arr):
	return max(range(len(arr)), key=lambda i: arr[i])

#used in creating the predictions files
def appendToFile(file, text):
	with open(file, 'a') as f:
	    f.write(text)

# bayes = NaiveBayes()




