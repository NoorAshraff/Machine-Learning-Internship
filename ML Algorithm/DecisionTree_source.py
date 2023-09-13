!pip install pydotplus
!pip install scikit-learn

# block 1
import pandas as pd # dataframe
import numpy as np # array
from sklearn import tree
import pydotplus

# block 2
# Generate a decision tree.
def createTree(trainingData):
    data = trainingData.iloc[:, :-1]  # Feature matrix
    labels = trainingData.iloc[:, -1]  # Labels
    trainedTree = tree.DecisionTreeClassifier(criterion="entropy")  # Decision tree classifier
    trainedTree.fit(data, labels)  # Train the model.
    return trainedTree

# block 3
def showtree2pdf(trainedTree,finename):
    dot_data = tree.export_graphviz(trainedTree, out_file=None) # Export the tree in Graphviz format.
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(finename)  # Save the tree diagram to the local machine in PDF format.

# block 4
def data2vectoc(data):
    names = data.columns[:-1]
    for i in names:
        col = pd.Categorical(data[i])
        data[i] = col.codes
    return data

# block 5
data = pd.read_table("tennis.txt",header=None,sep='\t') # Read training data.
trainingvec=data2vectoc(data) # Vectorize data.
decisionTree=createTree(trainingvec) # Create a decision tree.
showtree2pdf(decisionTree,"tennis.pdf") # Plot the decision tree.


dataset:   https://certification-data.obs.cn-north-4.myhuaweicloud.com/ENG/HCIA-AI/V3.5/chapter2/ML.zip


