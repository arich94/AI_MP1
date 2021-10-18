from sklearn import datasets as datasets
from sklearn import model_selection as mselect
from sklearn import naive_bayes as naive_bayes
from sklearn import metrics as metrics
from sklearn import tree as tree
from sklearn import linear_model as lm
from sklearn import neural_network as nn
import matplotlib.pyplot as plot
import sys
import numpy as np
import pandas as pandas

print("Starting Experiment... Please be patient...")

#part 2.2
csv = pandas.read_csv("drug200.csv")

#part 2.3
#making horizontal bar graph BBC-distribution.pdf
#Made with help from the matplotlib website
#https://matplotlib.org/stable/gallery/lines_bars_and_markers/barh.html#sphx-glr-gallery-lines-bars-and-markers-barh-py
plot.rcdefaults()
fig, graph = plot.subplots()

drugs = ('DrugA', 'DrugB', 'DrugC', 'DrugX', 'DrugY')
y_pos = np.arange(len(drugs))
instances = [len(csv[csv["Drug"]=="drugA"]), len(csv[csv["Drug"]=="drugB"]), len(csv[csv["Drug"]=="drugC"]), len(csv[csv["Drug"]=="drugX"]), len(csv[csv["Drug"]=="drugY"])]
error = np.random.rand(len(drugs))

graph.barh(y_pos, instances, xerr=error, align='center')
graph.set_yticks(y_pos)
graph.set_yticklabels(drugs)
graph.invert_yaxis()
graph.set_xlabel('Instances')
graph.set_title('Drug Distribution')

plot.savefig('drug-distribution.pdf', dpi=150)

#part 2.4
corpus = pandas.get_dummies(data=csv, columns=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])
corpus = corpus.drop(columns='Drug')

#part 2.5
x_corpus_train, x_corpus_test, y_corpus_train, y_corpus_test = mselect.train_test_split(corpus, csv["Drug"])

#part 2.8
fileName = "drug-performance.txt"
original_stdout = sys.stdout

nb_avg_macro = []
base_dt_avg_macro = []
top_dt_avg_macro = []
per_avg_macro = []
base_mlp_avg_macro = []
top_mlp_avg_macro = []

nb_avg_weight = []
base_dt_avg_weight = []
top_dt_avg_weight = []
per_avg_weight = []
base_mlp_avg_weight = []
top_mlp_avg_weight = []

nb_acc = []
base_dt_acc = []
top_dt_acc = []
per_acc = []
base_mlp_acc = []
top_mlp_acc = []

for i in range(10):
    print("Iteration Number: ", i+1)

    #part 2.6
    nb = naive_bayes.GaussianNB()
    base_dt = tree.DecisionTreeClassifier()
    temp_top_dt = tree.DecisionTreeClassifier()
    hyper_param_top_dt = {'criterion': ['gini', 'entropy'],
                        'max_depth': [5, 7],
                        'min_samples_split': [3, 5]}
    top_dt = mselect.GridSearchCV(estimator=temp_top_dt, param_grid=hyper_param_top_dt)
    per = lm.Perceptron()
    base_mlp = nn.MLPClassifier()
    temp_top_mlp = nn.MLPClassifier()
    hyper_param_top_mlp = {'activation': ['logistic', 'tanh', 'relu', 'identity'],
                        'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
                        'solver': ['adam', 'sgd']}
    top_mlp = mselect.GridSearchCV(estimator=temp_top_mlp, param_grid=hyper_param_top_mlp)

    nb.fit(x_corpus_train, y_corpus_train)
    base_dt.fit(x_corpus_train, y_corpus_train)
    top_dt.fit(x_corpus_train, y_corpus_train)
    per.fit(x_corpus_train, y_corpus_train)
    base_mlp.fit(x_corpus_train, y_corpus_train)
    top_mlp.fit(x_corpus_train, y_corpus_train)

    nb_test = nb.predict(x_corpus_test)
    base_dt_test = base_dt.predict(x_corpus_test)
    top_dt_test = top_dt.predict(x_corpus_test)
    per_test = per.predict(x_corpus_test)
    base_mlp_test = base_mlp.predict(x_corpus_test)
    top_mlp_test = top_mlp.predict(x_corpus_test)

    nb_metrics = metrics.classification_report(y_corpus_test, nb_test, output_dict=True)
    base_dt_metrics = metrics.classification_report(y_corpus_test, base_dt_test, output_dict=True)
    top_dt_metrics = metrics.classification_report(y_corpus_test, top_dt_test, output_dict=True)
    per_metrics = metrics.classification_report(y_corpus_test, per_test, output_dict=True)
    base_mlp_metrics = metrics.classification_report(y_corpus_test, base_mlp_test, output_dict=True)
    top_mlp_metrics = metrics.classification_report(y_corpus_test, top_mlp_test, output_dict=True)

    nb_avg_macro.append(nb_metrics['macro avg']['f1-score'])
    base_dt_avg_macro.append(base_dt_metrics['macro avg']['f1-score'])
    top_dt_avg_macro.append(top_dt_metrics['macro avg']['f1-score'])
    per_avg_macro.append(per_metrics['macro avg']['f1-score'])
    base_mlp_avg_macro.append(base_mlp_metrics['macro avg']['f1-score'])
    top_mlp_avg_macro.append(top_mlp_metrics['macro avg']['f1-score'])

    nb_avg_weight.append(nb_metrics['weighted avg']['f1-score'])
    base_dt_avg_weight.append(base_dt_metrics['weighted avg']['f1-score'])
    top_dt_avg_weight.append(top_dt_metrics['weighted avg']['f1-score'])
    per_avg_weight.append(per_metrics['weighted avg']['f1-score'])
    base_mlp_avg_weight.append(base_mlp_metrics['weighted avg']['f1-score'])
    top_mlp_avg_weight.append(top_mlp_metrics['weighted avg']['f1-score'])

    nb_acc.append(nb_metrics['accuracy'])
    base_dt_acc.append(base_dt_metrics['accuracy'])
    top_dt_acc.append(top_dt_metrics['accuracy'])
    per_acc.append(per_metrics['accuracy'])
    base_mlp_acc.append(base_mlp_metrics['accuracy'])
    top_mlp_acc.append(top_mlp_metrics['accuracy'])

    if i==0:
        file = open(fileName, "a")
        sys.stdout = file
        print("----------------------------------\na) Model: Gaussian Naive Bayes\nno hyper paramters changed\n")
        print("b) Confusion matrix:\n", metrics.confusion_matrix(nb_test, y_corpus_test), "\n")
        print("c) and d)\n", pandas.DataFrame(nb_metrics), "\n")

        print("----------------------------------\na) Model: Base Decision Tree\nno hyper paramters changed\n")
        print("b) Confusion matrix:\n", metrics.confusion_matrix(base_dt_test, y_corpus_test), "\n")
        print("c) and d)\n", pandas.DataFrame(base_dt_metrics), "\n")

        print("----------------------------------\na) Model: Top Decision Tree\nHyper Parameters changed through Grid Search: criterion, max depth, min smaples split\n")
        print("Best parameters chosen by Grid Search:\n", top_dt.best_params_, "\n")
        print("b) Confusion matrix:\n", metrics.confusion_matrix(top_dt_test, y_corpus_test), "\n")
        print("c) and d)\n", pandas.DataFrame(top_dt_metrics), "\n")

        print("----------------------------------\na) Model: Perceptron\nno hyper paramters changed\n")
        print("b) Confusion matrix:\n", metrics.confusion_matrix(per_test, y_corpus_test), "\n")
        print("c) and d)\n", pandas.DataFrame(per_metrics), "\n")

        print("----------------------------------\na) Model: Base Multi-Layer Perceptron\nNo hyper parameters changed\n")
        print("b) Confusion matrix:\n", metrics.confusion_matrix(base_mlp_test, y_corpus_test), "\n")
        print("c) and d)\n", pandas.DataFrame(base_mlp_metrics), "\n")

        print("----------------------------------\na) Model: Top Multi-Layer Perceptron\nHyper parameters changed through Grid Search: activation, hidden layer sizes, solver\n")
        print("Best parameters chosen by Grid Search:\n", pandas.DataFrame(top_mlp.best_params_), "\n")
        print("b) Confusion matrix:\n", metrics.confusion_matrix(top_mlp_test, y_corpus_test), "\n")
        print("c) and d)\n", pandas.DataFrame(top_mlp_metrics), "\n")
        sys.stdout = original_stdout
        file.close()

#calculate averages
nb_final_avg_acc = sum(nb_acc)/10
base_dt_final_avg_acc = sum(base_dt_acc)/10
top_dt_final_avg_acc = sum(top_dt_acc)/10
per_final_avg_acc = sum(per_acc)/10
base_mlp_final_avg_acc = sum(base_mlp_acc)/10
top_mlp_final_avg_acc = sum(top_mlp_acc)/10

nb_final_avg_macro = sum(nb_avg_macro)/10
base_dt_final_avg_macro = sum(base_dt_avg_macro)/10
top_dt_final_avg_macro = sum(top_dt_avg_macro)/10
per_final_avg_macro = sum(per_avg_macro)/10
base_mlp_final_avg_macro = sum(base_mlp_avg_macro)/10
top_mlp_final_avg_macro = sum(top_mlp_avg_macro)/10

nb_final_avg_weight = sum(nb_avg_weight)/10
base_dt_final_avg_weight = sum(base_dt_avg_weight)/10
top_dt_final_avg_weight = sum(top_dt_avg_weight)/10
per_final_avg_weight = sum(per_avg_weight)/10
base_mlp_final_avg_weight = sum(base_mlp_avg_weight)/10
top_mlp_final_avg_weight = sum(top_mlp_avg_weight)/10

#calculate standard deviations
nb_std_acc = np.std(nb_acc)
base_dt_std_acc = np.std(base_dt_acc)
top_dt_std_acc = np.std(top_dt_acc)
per_std_acc = np.std(per_acc)
base_mlp_std_acc = np.std(base_mlp_acc)
top_mlp_std_acc = np.std(top_mlp_acc)

nb_std_avg_macro = np.std(nb_avg_macro)
base_dt_std_avg_macro = np.std(base_dt_avg_macro)
top_dt_std_avg_macro = np.std(top_dt_avg_macro)
per_std_avg_macro = np.std(per_avg_macro)
base_mlp_std_avg_macro = np.std(base_mlp_avg_macro)
top_mlp_std_avg_macro = np.std(top_mlp_avg_macro)

nb_std_avg_weight = np.std(nb_avg_weight)
base_dt_std_avg_weight = np.std(base_dt_avg_weight)
top_dt_std_avg_weight = np.std(top_dt_avg_weight)
per_std_avg_weight = np.std(per_avg_weight)
base_mlp_std_avg_weight = np.std(base_mlp_avg_weight)
top_mlp_std_avg_weight = np.std(top_mlp_avg_weight)

mydata = {"Classifier": ["Gaussian Naive-Bayes", "Base Decision Tree", "Top Decision Tree", "Perceptron", "Base Multi-Layer Perceptron", "Top Multi-Layer Perceptron"],
          "Average Accuracy": [nb_final_avg_acc, base_dt_final_avg_acc, top_dt_final_avg_acc, per_final_avg_acc, base_mlp_final_avg_acc, top_mlp_final_avg_acc],
          "Average Macro Average F1": [nb_final_avg_macro, base_dt_final_avg_macro, top_dt_final_avg_macro, per_final_avg_macro, base_mlp_final_avg_macro, top_mlp_final_avg_macro],
          "Average Weighted Average F1": [nb_final_avg_weight, base_dt_final_avg_weight, top_dt_final_avg_weight, per_final_avg_weight, base_mlp_final_avg_weight, top_mlp_final_avg_weight],
          "Standard Deviation Accuracy": [nb_std_acc, base_dt_std_acc, top_dt_std_acc, per_std_acc, base_mlp_std_acc, top_mlp_std_acc],
          "Standard Deviation Macro Average F1": [nb_std_avg_macro, base_dt_std_avg_macro, top_dt_std_avg_macro, per_std_avg_macro, base_mlp_std_avg_macro, top_mlp_std_avg_macro],
          "Standard Deviation Weighted Average F1": [nb_std_avg_weight, base_dt_std_avg_weight, top_dt_std_avg_weight, per_std_avg_weight, base_mlp_std_avg_weight, top_mlp_std_avg_weight]}


file = open(fileName, "a")
sys.stdout = file
print(pandas.DataFrame(mydata))
sys.stdout = original_stdout
file.close()

print("Experiment finished.")