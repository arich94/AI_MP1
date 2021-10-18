import sklearn as sk
from sklearn import datasets as datasets
from sklearn import feature_extraction as fx
from sklearn import model_selection as mselect
from sklearn import naive_bayes as naive_bayes
from sklearn import preprocessing as preproc
from sklearn import metrics as metrics
import matplotlib.pyplot as plot
import sys
import numpy as np
import math

print("Starting experiment... please wait...")

#loading in files
corpus = datasets.load_files("BBC", encoding="latin1")

#making horizontal bar graph BBC-distribution.pdf
#Made with help from the matplotlib website
#https://matplotlib.org/stable/gallery/lines_bars_and_markers/barh.html#sphx-glr-gallery-lines-bars-and-markers-barh-py
plot.rcdefaults()
fig, graph = plot.subplots()

y_pos = np.arange(len(corpus.target_names))
instances = [510, 386, 417, 511, 401]
error = np.random.rand(len(corpus.target_names))

graph.barh(y_pos, instances, xerr=error, align='center')
graph.set_yticks(y_pos)
graph.set_yticklabels(corpus.target_names)
graph.invert_yaxis()
graph.set_xlabel('Instances')
graph.set_title('BBC Distribution')

plot.savefig('BBC-distribution.pdf', dpi=150)

#task 1.4 Pre-process dataset for Naive Bayes classifier using CountVectorizer
vectorizer = fx.text.CountVectorizer()
x_corpus = vectorizer.fit_transform(corpus.data)

encoder = preproc.LabelEncoder()
y_corpus = encoder.fit_transform(corpus.target)

#for task 1.5 split dataset
x_corpus_train, x_corpus_test, y_corpus_train, y_corpus_test = mselect.train_test_split(x_corpus, y_corpus, train_size=0.8, test_size=0.2, random_state=None)

#for task 1.6 create MultinomialNB and train data set
clf = naive_bayes.MultinomialNB()
clf.fit(x_corpus_train, y_corpus_train)
firstTest = clf.predict(x_corpus_test)

#for task 1.7 writing to file
sample = naive_bayes.MultinomialNB()
sample.fit(x_corpus, y_corpus)

#open file for appending results. Comment out the following two lines to print to console.
original_stdout = sys.stdout
file = open("bbc-performance.txt", "a")
sys.stdout = file

print("------------------------------\na) MultinomialNB default values, try 1\n")
print("b) - Confusion Matrix\n", metrics.confusion_matrix(firstTest, y_corpus_test), "\n")
print("c)", metrics.classification_report(y_corpus_test, firstTest, target_names=corpus.target_names))
print("d) - Prior probability of each class:\n     business - 0.23\nentertainment - 0.17\n     politics - 0.19\n        sport - 0.23\n         tech - 0.18")
print("f) - Size of vocabulary:\n", len(vectorizer.vocabulary_), "\n")

business=0
for cat in x_corpus[corpus.target==0].toarray():
    for f in cat:
        business += f
entertainment=0
for cat in x_corpus[corpus.target==1].toarray():
    for f in cat:
        entertainment += f
politics=0
for cat in x_corpus[corpus.target==2].toarray():
    for f in cat:
        politics += f
sport=0
for cat in x_corpus[corpus.target==3].toarray():
    for f in cat:
        sport += f
tech=0
for cat in x_corpus[corpus.target==4].toarray():
    for f in cat:
        tech += f

print("g) - Number of word-tokens in each class:\n     business - ", business, "\nentertainment - ", entertainment, "\n     politics - ", politics, "\n        sport - ", sport, "\n         tech - ", tech, "\n")
print("h) - Number of word-tokens in the entire corpus:\n", business+entertainment+politics+sport+tech, "\n")

business=0
for cat in x_corpus[corpus.target==0].toarray():
    for f in cat:    
        if f==0:
            business += 1
entertainment=0
for cat in x_corpus[corpus.target==1].toarray():
    for f in cat:
        if f==0:
            entertainment += 1
politics=0
for cat in x_corpus[corpus.target==2].toarray():
    for f in cat:
        if f==0:
            politics += 1
sport=0
for cat in x_corpus[corpus.target==3].toarray():
    for f in cat:
        if f==0:
            sport += 1
tech=0
for cat in x_corpus[corpus.target==4].toarray():
    for f in cat:
        if f==0:
            tech += 1

print("i) - Number of word with a frequency of 0 in each class:\n     business - ", business, "\nentertainment - ", entertainment, "\n     politics - ", politics, "\n        sport - ", sport, "\n         tech - ", tech)
print("Percentages do not make sense. These values seem off.\n")
business=0
for cat in x_corpus[corpus.target==0].toarray():
    for f in cat:    
        if f==1:
            business += 1
entertainment=0
for cat in x_corpus[corpus.target==1].toarray():
    for f in cat:
        if f==1:
            entertainment += 1
politics=0
for cat in x_corpus[corpus.target==2].toarray():
    for f in cat:
        if f==1:
            politics += 1
sport=0
for cat in x_corpus[corpus.target==3].toarray():
    for f in cat:
        if f==1:
            sport += 1
tech=0
for cat in x_corpus[corpus.target==4].toarray():
    for f in cat:
        if f==1:
            tech += 1

print("j) - Number of words in the entire corpus with a frequency of 1:\n", business+entertainment+politics+sport+tech, "(38.24%)\n")

wordOne = sample.feature_count_[0][5] + sample.feature_count_[1][5] + sample.feature_count_[2][5] + sample.feature_count_[3][5] + sample.feature_count_[4][5]
wordTwo = sample.feature_count_[0][37] + sample.feature_count_[1][37] + sample.feature_count_[2][37] + sample.feature_count_[3][37] + sample.feature_count_[4][37]

wordOneLogProb = math.log(wordOne/836357)
wordTwoLogProb = math.log(wordTwo/836357)
print("k) Log probability of two given words:\nWord One: ", wordOneLogProb, "\nWord Two: ", wordTwoLogProb)

#for task 1.8, repeating tasks 1.6 and 1.7
#for task 1.6 create MultinomialNB and train data set
clf = naive_bayes.MultinomialNB()
clf.fit(x_corpus_train, y_corpus_train)
firstTest = clf.predict(x_corpus_test)

#for task 1.7 writing to file
sample = naive_bayes.MultinomialNB()
sample.fit(x_corpus, y_corpus)

print("------------------------------\na) MultinomialNB default values, try 2\n")
print("b) - Confusion Matrix\n", metrics.confusion_matrix(firstTest, y_corpus_test), "\n")
print("c)", metrics.classification_report(y_corpus_test, firstTest, target_names=corpus.target_names))
print("d) - Prior probability of each class:\n     business - 0.23\nentertainment - 0.17\n     politics - 0.19\n        sport - 0.23\n         tech - 0.18")
print("f) - Size of vocabulary:\n", len(vectorizer.vocabulary_), "\n")

business=0
for cat in x_corpus[corpus.target==0].toarray():
    for f in cat:
        business += f
entertainment=0
for cat in x_corpus[corpus.target==1].toarray():
    for f in cat:
        entertainment += f
politics=0
for cat in x_corpus[corpus.target==2].toarray():
    for f in cat:
        politics += f
sport=0
for cat in x_corpus[corpus.target==3].toarray():
    for f in cat:
        sport += f
tech=0
for cat in x_corpus[corpus.target==4].toarray():
    for f in cat:
        tech += f

print("g) - Number of word-tokens in each class:\n     business - ", business, "\nentertainment - ", entertainment, "\n     politics - ", politics, "\n        sport - ", sport, "\n         tech - ", tech, "\n")
print("h) - Number of word-tokens in the entire corpus:\n", business+entertainment+politics+sport+tech, "\n")

business=0
for cat in x_corpus[corpus.target==0].toarray():
    for f in cat:    
        if f==0:
            business += 1
entertainment=0
for cat in x_corpus[corpus.target==1].toarray():
    for f in cat:
        if f==0:
            entertainment += 1
politics=0
for cat in x_corpus[corpus.target==2].toarray():
    for f in cat:
        if f==0:
            politics += 1
sport=0
for cat in x_corpus[corpus.target==3].toarray():
    for f in cat:
        if f==0:
            sport += 1
tech=0
for cat in x_corpus[corpus.target==4].toarray():
    for f in cat:
        if f==0:
            tech += 1

print("i) - Number of word with a frequency of 0 in each class:\n     business - ", business, "\nentertainment - ", entertainment, "\n     politics - ", politics, "\n        sport - ", sport, "\n         tech - ", tech)
print("Percentages do not make sense. These values seem off.\n")
business=0
for cat in x_corpus[corpus.target==0].toarray():
    for f in cat:    
        if f==1:
            business += 1
entertainment=0
for cat in x_corpus[corpus.target==1].toarray():
    for f in cat:
        if f==1:
            entertainment += 1
politics=0
for cat in x_corpus[corpus.target==2].toarray():
    for f in cat:
        if f==1:
            politics += 1
sport=0
for cat in x_corpus[corpus.target==3].toarray():
    for f in cat:
        if f==1:
            sport += 1
tech=0
for cat in x_corpus[corpus.target==4].toarray():
    for f in cat:
        if f==1:
            tech += 1

print("j) - Number of words in the entire corpus with a frequency of 1:\n", business+entertainment+politics+sport+tech, "(38.24%)\n")

wordOne = sample.feature_count_[0][5] + sample.feature_count_[1][5] + sample.feature_count_[2][5] + sample.feature_count_[3][5] + sample.feature_count_[4][5]
wordTwo = sample.feature_count_[0][37] + sample.feature_count_[1][37] + sample.feature_count_[2][37] + sample.feature_count_[3][37] + sample.feature_count_[4][37]

wordOneLogProb = math.log(wordOne/836357)
wordTwoLogProb = math.log(wordTwo/836357)
print("k) Log probability of two given words:\nWord One: ", wordOneLogProb, "\nWord Two: ", wordTwoLogProb)


#repeating again for task 1.9 -- this time with smoothing
#for task 1.6 create MultinomialNB and train data set
clf = naive_bayes.MultinomialNB(alpha=0.0001)
clf.fit(x_corpus_train, y_corpus_train)
firstTest = clf.predict(x_corpus_test)

#for task 1.7 writing to file
sample = naive_bayes.MultinomialNB(alpha=0.0001)
sample.fit(x_corpus, y_corpus)

print("------------------------------\na) MultinomialNB smoothing=0.0001, try 3\n")
print("b) - Confusion Matrix\n", metrics.confusion_matrix(firstTest, y_corpus_test), "\n")
print("c)", metrics.classification_report(y_corpus_test, firstTest, target_names=corpus.target_names))
print("d) - Prior probability of each class:\n     business - 0.23\nentertainment - 0.17\n     politics - 0.19\n        sport - 0.23\n         tech - 0.18")
print("f) - Size of vocabulary:\n", len(vectorizer.vocabulary_), "\n")

business=0
for cat in x_corpus[corpus.target==0].toarray():
    for f in cat:
        business += f
entertainment=0
for cat in x_corpus[corpus.target==1].toarray():
    for f in cat:
        entertainment += f
politics=0
for cat in x_corpus[corpus.target==2].toarray():
    for f in cat:
        politics += f
sport=0
for cat in x_corpus[corpus.target==3].toarray():
    for f in cat:
        sport += f
tech=0
for cat in x_corpus[corpus.target==4].toarray():
    for f in cat:
        tech += f

print("g) - Number of word-tokens in each class:\n     business - ", business, "\nentertainment - ", entertainment, "\n     politics - ", politics, "\n        sport - ", sport, "\n         tech - ", tech, "\n")
print("h) - Number of word-tokens in the entire corpus:\n", business+entertainment+politics+sport+tech, "\n")

business=0
for cat in x_corpus[corpus.target==0].toarray():
    for f in cat:    
        if f==0:
            business += 1
entertainment=0
for cat in x_corpus[corpus.target==1].toarray():
    for f in cat:
        if f==0:
            entertainment += 1
politics=0
for cat in x_corpus[corpus.target==2].toarray():
    for f in cat:
        if f==0:
            politics += 1
sport=0
for cat in x_corpus[corpus.target==3].toarray():
    for f in cat:
        if f==0:
            sport += 1
tech=0
for cat in x_corpus[corpus.target==4].toarray():
    for f in cat:
        if f==0:
            tech += 1

print("i) - Number of word with a frequency of 0 in each class:\n     business - ", business, "\nentertainment - ", entertainment, "\n     politics - ", politics, "\n        sport - ", sport, "\n         tech - ", tech)
print("Percentages do not make sense. These values seem off.\n")
business=0
for cat in x_corpus[corpus.target==0].toarray():
    for f in cat:    
        if f==1:
            business += 1
entertainment=0
for cat in x_corpus[corpus.target==1].toarray():
    for f in cat:
        if f==1:
            entertainment += 1
politics=0
for cat in x_corpus[corpus.target==2].toarray():
    for f in cat:
        if f==1:
            politics += 1
sport=0
for cat in x_corpus[corpus.target==3].toarray():
    for f in cat:
        if f==1:
            sport += 1
tech=0
for cat in x_corpus[corpus.target==4].toarray():
    for f in cat:
        if f==1:
            tech += 1

print("j) - Number of words in the entire corpus with a frequency of 1:\n", business+entertainment+politics+sport+tech, "(38.24%)\n")

wordOne = sample.feature_count_[0][5] + sample.feature_count_[1][5] + sample.feature_count_[2][5] + sample.feature_count_[3][5] + sample.feature_count_[4][5]
wordTwo = sample.feature_count_[0][37] + sample.feature_count_[1][37] + sample.feature_count_[2][37] + sample.feature_count_[3][37] + sample.feature_count_[4][37]

wordOneLogProb = math.log(wordOne/836357)
wordTwoLogProb = math.log(wordTwo/836357)
print("k) Log probability of two given words:\nWord One: ", wordOneLogProb, "\nWord Two: ", wordTwoLogProb)


#repeating again for task 1.9 -- this time with smoothing
#for task 1.6 create MultinomialNB and train data set
clf = naive_bayes.MultinomialNB(alpha=0.9)
clf.fit(x_corpus_train, y_corpus_train)
firstTest = clf.predict(x_corpus_test)

#for task 1.7 writing to file
sample = naive_bayes.MultinomialNB(alpha=0.9)
sample.fit(x_corpus, y_corpus)

print("------------------------------\na) MultinomialNB smoothing = 0.9, try 4\n")
print("b) - Confusion Matrix\n", metrics.confusion_matrix(firstTest, y_corpus_test), "\n")
print("c)", metrics.classification_report(y_corpus_test, firstTest, target_names=corpus.target_names))
print("d) - Prior probability of each class:\n     business - 0.23\nentertainment - 0.17\n     politics - 0.19\n        sport - 0.23\n         tech - 0.18")
print("f) - Size of vocabulary:\n", len(vectorizer.vocabulary_), "\n")

business=0
for cat in x_corpus[corpus.target==0].toarray():
    for f in cat:
        business += f
entertainment=0
for cat in x_corpus[corpus.target==1].toarray():
    for f in cat:
        entertainment += f
politics=0
for cat in x_corpus[corpus.target==2].toarray():
    for f in cat:
        politics += f
sport=0
for cat in x_corpus[corpus.target==3].toarray():
    for f in cat:
        sport += f
tech=0
for cat in x_corpus[corpus.target==4].toarray():
    for f in cat:
        tech += f

print("g) - Number of word-tokens in each class:\n     business - ", business, "\nentertainment - ", entertainment, "\n     politics - ", politics, "\n        sport - ", sport, "\n         tech - ", tech, "\n")
print("h) - Number of word-tokens in the entire corpus:\n", business+entertainment+politics+sport+tech, "\n")

business=0
for cat in x_corpus[corpus.target==0].toarray():
    for f in cat:    
        if f==0:
            business += 1
entertainment=0
for cat in x_corpus[corpus.target==1].toarray():
    for f in cat:
        if f==0:
            entertainment += 1
politics=0
for cat in x_corpus[corpus.target==2].toarray():
    for f in cat:
        if f==0:
            politics += 1
sport=0
for cat in x_corpus[corpus.target==3].toarray():
    for f in cat:
        if f==0:
            sport += 1
tech=0
for cat in x_corpus[corpus.target==4].toarray():
    for f in cat:
        if f==0:
            tech += 1

print("i) - Number of word with a frequency of 0 in each class:\n     business - ", business, "\nentertainment - ", entertainment, "\n     politics - ", politics, "\n        sport - ", sport, "\n         tech - ", tech)
print("Percentages do not make sense. These values seem off.\n")
business=0
for cat in x_corpus[corpus.target==0].toarray():
    for f in cat:    
        if f==1:
            business += 1
entertainment=0
for cat in x_corpus[corpus.target==1].toarray():
    for f in cat:
        if f==1:
            entertainment += 1
politics=0
for cat in x_corpus[corpus.target==2].toarray():
    for f in cat:
        if f==1:
            politics += 1
sport=0
for cat in x_corpus[corpus.target==3].toarray():
    for f in cat:
        if f==1:
            sport += 1
tech=0
for cat in x_corpus[corpus.target==4].toarray():
    for f in cat:
        if f==1:
            tech += 1

print("j) - Number of words in the entire corpus with a frequency of 1:\n", business+entertainment+politics+sport+tech, "(38.24%)\n")

wordOne = sample.feature_count_[0][5] + sample.feature_count_[1][5] + sample.feature_count_[2][5] + sample.feature_count_[3][5] + sample.feature_count_[4][5]
wordTwo = sample.feature_count_[0][37] + sample.feature_count_[1][37] + sample.feature_count_[2][37] + sample.feature_count_[3][37] + sample.feature_count_[4][37]

wordOneLogProb = math.log(wordOne/836357)
wordTwoLogProb = math.log(wordTwo/836357)
print("k) Log probability of two given words:\nWord One: ", wordOneLogProb, "\nWord Two: ", wordTwoLogProb)

file.close()
sys.stdout = original_stdout

print("Experiment complete.")