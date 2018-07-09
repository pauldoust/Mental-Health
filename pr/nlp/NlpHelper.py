import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import ngrams
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import unicodedata
from nltk.stem.isri import ISRIStemmer
ps = PorterStemmer()
st = ISRIStemmer()
lemma = WordNetLemmatizer()
exclude = set(string.punctuation)
# bulletPattern = re.compile(r'[•|o|*] \s+ ([A-Z]+)')
bulletPattern = re.compile(r'[•|\*|o](\s+)([A-Z]+)')


class PreProcessing:

    # Perfom tokenization (by word or sentence) on an input sentence
    @staticmethod
    def tokenize(sentence, tokenizing="word"):
        sentence = sentence.lower()
        if(tokenizing == "word"):
            return word_tokenize(sentence)
        else:
            return sent_tokenize(sentence)

    # Prform filtering on a sentence using default stopwords for each language (default english)
    # customStopWords (optional): custom set of user defined stop words
    @staticmethod
    def filterStopWords(sentence, customStopWords=[], lang="english"):
        stop_words = set(stopwords.words(lang))
        filtered_sentence = []

        if not isinstance(sentence, str):
            return filtered_sentence
        tokenizedWords = PreProcessing.tokenize(sentence)
        filtered_sentence = [
            w for w in tokenizedWords if w not in stop_words and w not in customStopWords]
        return filtered_sentence

    # s Perform Stemming for the inpur word
    @staticmethod
    def stem(wordsToStem):
        if isinstance(wordsToStem, list):
            # stemmed_words = [ps.stem(w) for w in wordsToStem]
            stemmed_words = [st.stem(w) for w in wordsToStem]
            return stemmed_words
        elif isinstance(wordsToStem, str):
            # print(wordsToStem)
            return st.stem(wordsToStem)
            # return ps.stem(wordsToStem)

    # Perform pos tagging for an input sentence
    @staticmethod
    def posTagging(sentence):
        try:
            tokenizedWords = PreProcessing.tokenize(sentence)
            taggedTokens = nltk.pos_tag(tokenizedWords)
            # print(taggedTokens)
            return taggedTokens
        except Exception as e:
            print(str(e))

    # Perform Chunking according to grammar and a posTagged sentence
    @staticmethod
    def chunk(taggedSentence, grammar):
        cp = nltk.RegexpParser(grammar)
        result = cp.parse(taggedSentence)
        # result.draw()
        return result

    @staticmethod
    def shallowParsing(sentence, customToExclude=[], thershold = 4):
        if isinstance(sentence, float):
            return ""
        # sentence = ' '.join([i for i in sentence.split() if i not in customToExclude])
        # print(sentence)
        # print(sentence.split("."))
        # return
        # sentence = unicodedata.normalize("NFKD", sentence)
        # sentence.replace('\n'," ")
        sentence = re.sub(bulletPattern,r'. \2',sentence)
        # print(sentence)
        res = []
        sentences = sentence.split(".")
        # print(sentences)

       	grammar = r"""
	            NP: {<NN.*|JJ>*<NN.*>}
	            J:  {<JJ*|RB*>+}
	                {<JJ*>+<TO><V*>} 
	            NP: {<NN.*><IN><NN.*>}
	            NR: {<NP>}
	                {<NP><IN><NP>}
	        """
        for s in sentences:
        	if not s:
        		continue
        	# print(s)
        	s = s.replace('\n',"")
        	if len(s.split()) <= thershold:
        		res.append(s)
        		continue
        	taggedSent = PreProcessing.posTagging(s)
        	chunked = PreProcessing.chunk(taggedSent, grammar)
        	result = PreProcessing.extract_entity(chunked, "NR")
        	res.append(result)
	       # break

        return res

    @staticmethod
    def ngrams(text, n=2):
        tokenizedWords = PreProcessing.tokenize(text)
        grams = ngrams(tokenizedWords, n)
        return grams

    @staticmethod
    def countFrequency(ngrm, top=1):
        fdist = nltk.FreqDist(ngrm)
        return fdist.most_common(top)

    @staticmethod
    def clean(doc, customStopWords=[], withStemming=False, lang="english"):
        if isinstance(doc, float):
            return ""
        tokeniziedWords = PreProcessing.tokenize(doc)
        # customStopWords = customStopWords.append(exclude)
        # stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        stop_free = " ".join(PreProcessing.filterStopWords(
            " ".join(tokeniziedWords), customStopWords, lang = lang))
        # punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        punc_free = ''.join(
            ch if ch not in exclude else ' ' for ch in stop_free)
        normalized = " ".join(lemma.lemmatize(word)
                              for word in punc_free.split())
        if(withStemming):
            stemmed = " ".join(PreProcessing.stem(normalizedWord)
                               for normalizedWord in normalized.split())
            return stemmed
        else:
            return normalized

    @staticmethod
    def leaves(tree, tag):
        for subtree in tree.subtrees(filter=lambda t: hasattr(t, 'label') and t.label() == tag):
            yield subtree.leaves()

    @staticmethod
    def extract_entity(tree, tag):
        roles = []
        # merge these phrases and store them as a possible role
        for leaf in PreProcessing.leaves(tree, tag):
            term = [w for w, t in leaf]
            roles.append(" ".join(term))
        return roles


# print(PreProcessing.shallowParsing("hello Computer Science"))
# test = PreProcessing.shallowParsing("hello Computer Science")
# print(extract_entity(test,"NR"))
# # test = PreProcessing.shallowParsing("hello Computer Science")
# # for r in test:
# #     if type(r) == nltk.tree.Tree:
# #         print('NP:', ' '.join([x[0] for x in r.leaves()]))


# s = "o Fello world. how are you ? are you doing fine !!"
# print(re.sub(r'o ([A-Z]+)',r'. \1',s))

# text = """• Cancer diagnostic
# • Anticancer therapeutic"""
# # text = """. Cancer diagnostic 
# #   . Anticancer therapeutic"""
# pattern = re.compile(r'[\*|o|•] \s+ ([A-Z]+)')
# # text = pattern.sub('',r'. \1',text)
# text = re.sub(r'[•|o|*] ([A-Z]+)',r'. \1',text)
# # print(text)

# # print(PreProcessing.shallowParsing(text))


# t = """ o      In-situ fresh preparation of final nanopore devices with reproducible properties o  Start-kit chips are provided to users, which can be massively produced with standard top-down lithographic processes
# o   Immediately before recording, automated feedback-controlled process is used to “finish” the nanopore so that the dimension and geometry of each device can be evaluated and optimized in real-time
# •       Self-aligned control electrodes for precise translocation control and readout
# o       Act as an active gate. Much stronger control over molecule-nanopore interactions by time-modulated transverse bias.
# o       DNA can be “stepped” through the nanopore to achieve greater precision in position control and stability in readout signals
# o       Recognition tunneling current across the electrodes provides higher spatial resolution and specificity, can be correlated with simultaneous ionic current readout
# •       Unique planar chip architecture allows for simultaneous optical access of each device for real-time fluorescence and Raman studies
# """.strip()
# # t = t.replace("\n"," ")
# new_str = unicodedata.normalize("NFKD", t)
# # print([new_str])
# # print([t])
# # pattern = re.compile(r'\no\xa0\xa0\xa0\xa0\xa0\xa0\xa0')
# pattern = re.compile(r'[•|o|*] \s+ ([A-Z]+)')
# text = re.sub(pattern,r'. \1',new_str)
# # text = re.sub(,r'. ',t)
# print(text)

# print(re.sub(r'[\*|o|•] ([A-Z]+)',r'. \1',t))
# print(PreProcessing.shallowParsing(re.sub(r'[\*|o|•] ([A-Z]+)',r'. \1',t)))

# t = """•       In-situ fresh preparation of final nanopore devices with reproducible properties
# o       Start-kit chips are provided to users, which can be massively produced with standard top-down lithographic processes
# o       Immediately before recording, automated feedback-controlled process is used to “finish” the nanopore so that the dimension and geometry of each device can be evaluated and optimized in real-time
# •       Self-aligned control electrodes for precise translocation control and readout
# o       Act as an active gate. Much stronger control over molecule-nanopore interactions by time-modulated transverse bias.
# o       DNA can be “stepped” through the nanopore to achieve greater precision in position control and stability in readout signals
# o       Recognition tunneling current across the electrodes provides higher spatial resolution and specificity, can be correlated with simultaneous ionic current readout
# •       Unique planar chip architecture allows for simultaneous optical access of each device for real-time fluorescence and Raman studies"""
# new_str = unicodedata.normalize("NFKD", t)
# print(PreProcessing.shallowParsing(t))


# t="""• Provides a homogeneous coupling of molecular monomers to at least 95% of the surface
# • Consistent production with high quality
# • Temperature control/improved level of heat distribution - provides a level of thermal uniformity across a surface that varies by no more than 5oC
# • Can effectively distribute the amount of fluid comprising the monomers
# • Allows for the removal of a used solution or for the replacement of a solution that is no longer needed
# • Enables effective diffusion of fresh solution to the surface – speeds up the reaction and removes unwanted reactive species"""

# print([t])
# s = re.compile(r'[•|\*|o](\s+)([A-Z]+)')
# print(s)
# print(re.sub(s,r'. \2',t))
# print(PreProcessing.shallowParsing(t))

# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# x = np.array([2,3,1,0])
# y = np.array([2,3,0.9,0])

# print(cosine_similarity(x,y)) # = array([[ 0.96362411]]), most similar

# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np


# title1 = np.array([2,3,1.1,0])
# title2 = np.array([2,3,1,0.2])
# title3 = np.array([2,3,1,1])
# title4 = np.array([2,3,1,1])
# title5 = np.array([2,13,12,10])

# # titles = [title1, title2, title3, title4, title5]
# titles = np.stack((title1,title2))
# # print(titles)
# # print(type(titles))
# N = len(titles)


# similarity_matrix = np.array(N**1*[0],np.float).reshape(1,N)

# # for m in range(N):
# #     similarity_matrix [m,m] = 1
# #     for n in range(m+1,N):
# for n in range(N):
# 	m = 0
# 	print("comp:")
# 	print(n)
# 	print(title1)
# 	print(titles[n])
# 	similarity_matrix [m,n] = cosine_similarity(title1, titles[n])
# 	# similarity_matrix [m,n] = cosine_similarity(titles[m], titles[n])
# 	# similarity_matrix [n,m] = similarity_matrix [m,n]
# 	similarity_matrix [m,m] = -100

# print (similarity_matrix)
# f = list(range(1, 10 + 1))
# print(f)
# topics_no =500
# f = ['Topic ' + str(i) for i in range(1,topics_no + 1)]
# print(f)
# import nltk

# w= 'حركات'
# print(st.stem(w))