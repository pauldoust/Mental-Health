from pr.nlp.NlpHelper import * 
from pr.dataloading.DataHandler import * 
import gensim
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from gensim import corpora
import pandas as pd
import re
import heapq
import random
import csv
def main():
	topics_no = 5
	datasetFrames = []
	ldamodel = loadLdaModel('lda.model', num_topics = topics_no, passes = 100)
	topics = ldamodel.print_topics(num_topics=topics_no , num_words=10) 
	for t in topics:
		topic_terms = ldamodel.get_topic_terms(t[0])
		topic_words = " ".join([ldamodel.id2word[term] for term,prob in topic_terms])
		print("Topic #" + str(t[0] + 1) + ": ", topic_words)

	with open('dict.txt', 'w') as csv_file:
		for key, value in ldamodel.id2word.items():
			csv_file.write(str(key) + '\t' + str(value) + '\n')
	print(ldamodel.id2word.doc2bow(["acknowledge", "acknowledge", "zoroarks"]))


	datasetFrames = ProcessDataset("processedData.xlsx")
	columns = ['Topic ' + str(i)  for i in range(1,topics_no + 1)]
	bow_features_columns = [str(key)   for key,value in ldamodel.id2word.items()]
	columns = columns + bow_features_columns
	# probArray = pd.DataFrame(columns = columns )
	probArray = pd.DataFrame(columns = columns)
	if FilesHandler.ifFileExists("TopicDoc.xlsx"):
		print("Probability Distribution found, loading..")
		distFile = ExcelHandler.loadFromDirectory(".", format="xlsx", isDebug=True, specificFileName="TopicDoc.xlsx" )
		distFrames = ExcelHandler.loadIntoDataframe(distFile)
	else:
		print("Probability Distribution not found, creating..")
		topics_features = {'Topic ' + str(i)  : str(0) for i in range(1,topics_no + 1)}
		bow_features = { str(key)    :str(0) for key, value in ldamodel.id2word.items()}
		f = dict(topics_features)
		f.update(bow_features) 
		for index, row in datasetFrames.iterrows():
			print("index: ",index)
			# if isinstance(row["Summary"],float):
				# continue
			docBow = ldamodel.id2word.doc2bow(row["Summary"].split())
			for top,pro in ldamodel[docBow]:
				f['Topic '+ str(top+1)] = pro
			for k,v in docBow:
				f[str(k)] = v
			print(index)
			if(index > 1000 ):
				continue
			probArray.loc[index] = f		
		
		# pd.concat([f ], ignore_index)	
		# probArray["topic"] = datasetFrames["topic"]
		print("finish 1")
		writer = pd.ExcelWriter("TopicDoc.xlsx")
		print("finish 2")
		probArray.to_excel(writer,'Sheet1')
		writer.close()

	docs = []
	

	return

def loadLdaModel(modelName, num_topics = 10, passes= 1):
	if FilesHandler.ifFileExists(modelName):
		print("Loading Trained Model...")
		ldamodel =  gensim.models.LdaModel.load(modelName)
		print("Model Loaded")
		print(ldamodel.id2word)
		print("Extracted Latent Topics: ")
	else:
		print("Model not found, training..")
		datasetFrames = ProcessDataset("processedData.xlsx")
		doc_complete  = datasetFrames["Summary"].as_matrix()
		doc_clean = [PreProcessing.clean(doc).split() for doc in doc_complete]
		dictionary = corpora.Dictionary(doc_clean)
		doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
		Lda = gensim.models.ldamodel.LdaModel
		ldamodel = Lda(doc_term_matrix, num_topics=num_topics, id2word = dictionary, passes=passes)
		# topics = ldamodel.print_topics(num_topics=50, num_words=10) 
		ldamodel.save('lda.model')
		print("Model trained and saved successfully")
	return ldamodel

def ProcessDataset(datasetName):
	if FilesHandler.ifFileExists(datasetName):
		print("Pre-Processed Dataset found, Loading..")
		dataFiles = ExcelHandler.loadFromDirectory(".", format="xlsx", isDebug=True, specificFileName=datasetName )
		datasetFrames =ExcelHandler.loadIntoDataframe(dataFiles)
	else:
		print("Pre-Processed Dataset not found, Processing..")

		dataFiles = ExcelHandler.loadFromDirectory("Dataset/", format="", isDebug=True)
		datasetFrames =ExcelHandler.loadIntoDataframe(dataFiles)

		datasetFrames = datasetFrames.replace(np.nan, '', regex=True)
		# datasetFramesToProcess = datasetFrames.copy()
		datasetFramesToProcess = pd.DataFrame()

		# datasetFramesToProcess['Summary'] = pd.Series(datasetFramesToProcess.fillna(' ').values.tolist()).str.join(' ')
		datasetFramesToProcess['text'] = datasetFrames['text']
		datasetFramesToProcess['topic'] = datasetFrames['topic']
		datasetFramesToProcess['user_friends_count'] = datasetFrames['user_friends_count']
		datasetFramesToProcess['Summary'] = datasetFrames['text'].apply(lambda x:  PreProcessing.clean(x,customStopWords = ["*","•","«","»","،","،،"], withStemming=True))

		# datasetFrames['Summary'] = pd.Series(datasetFrames.fillna('').values.tolist()).str.join(' ')
		# datasetFrames['Summary'] = datasetFrames['Summary'].apply(lambda x:  PreProcessing.clean(x,customStopWords = ["*","•"]))
		datasetFrames['Summary'] = datasetFramesToProcess['Summary']
		writer = pd.ExcelWriter(datasetName)
		datasetFramesToProcess.to_excel(writer,'Sheet1')
		print("Dataset is processed and saved in the same directory")

	print("Data Description: ")
	ExcelHandler.describe(datasetFrames)
	return datasetFrames




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
def kmean(mydocs, clusters_num):
	# mydocs = ["Human machine interface for lab abc computer applications", "A survey of user opinion of computer system response time", "The EPS user interface management system", "System and human system engineering testing of EPS", "Relation of user perceived response time to error measurement", "The generation of random binary unordered trees", "The intersection graph of paths in trees", "Graph minors IV Widths of trees and well quasi ordering", "Graph minors A survey"]
	vectorizer = TfidfVectorizer(stop_words='english')
	X = vectorizer.fit_transform(mydocs)
	true_k = clusters_num
	model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1)
	model.fit(X)
	print("Top terms per cluster:")
	order_centroids = model.cluster_centers_.argsort()[:, ::-1]
	terms = vectorizer.get_feature_names()
	for i in range(true_k):
		print ("Cluster %d: " % i)
		w=""
		for ind in order_centroids[i, :]:
			w = w + " " + terms[ind]
			# print (' ' + terms[ind])
		print(w)
	return order_centroids

def wardClustering(mydocs):
	if FilesHandler.ifFileExists("linkage_matrix.dat"):
		linkage_matrix = np.load("linkage_matrix.dat")
		print(linkage_matrix)
	else:
		print("ward clustering")
		vectorizer = TfidfVectorizer(stop_words='english')
		X = vectorizer.fit_transform(mydocs)
		# print("docs: ")
		# print(mydocs)
		# print("X")		
		# print(X)
		dist = 1 - cosine_similarity(X)
		# print("dist")
		# print(dist)
		linkage_matrix = ward(dist)
		# print("linksage")
		# print(linkage_matrix)

		linkage_matrix.dump("linkage_matrix.dat")

	# print(type(linkage_matrix))
	# print(linkage_matrix.shape)
	# print(linkage_matrix[5,:])
	fig, ax = plt.subplots()
	ax = dendrogram(linkage_matrix, orientation="right");
	plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')
	plt.tight_layout()
	plt.savefig('ward_clusters.png', dpi=1024) #save figure as ward_clusters
	plt.close()
	print("Clustering Ended")
	return linkage_matrix

main()
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
data_corpus = ["John likes to watch movies. Mary likes movies too.", 
"John also likes to watch football games."]
X = vectorizer.fit_transform(data_corpus) 
print(X.toarray())
print(vectorizer.get_feature_names())
print(type(X))