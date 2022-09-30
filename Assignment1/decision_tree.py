# -------------------------------------------------------------------------
# AUTHOR: Yi Huang
# FILENAME: decision_tree.py
# SPECIFICATION: Implement decision tree algorithm ID3
# FOR: CS 4210- Assignment #1
# TIME SPENT: 1 hour 30 minutes
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv

db = []
X = []
Y = []

# reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)
            print(row)


# transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
# --> add your Python code here

# The converted number of each features value
Age = {"Young": 1, "Presbyopic": 2, "Prepresbyopic": 3}
Spectacle = {"Myope": 1, "Hypermetrope": 2}
Astigmatism = {"Yes": 1, "No": 2}
Tear_Production_Rate = {"Normal": 1, "Reduced": 2}
for instance in db:
    temp = [Age[instance[0]], Spectacle[instance[1]], Astigmatism[instance[2]], Tear_Production_Rate[instance[3]]]
    X.append(temp)
# print(X)


# transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
# --> addd your Python code here
categorical_training_classes = {"Yes": 1, "No": 2}
for instance in db:
    Y.append(categorical_training_classes[instance[4]])
# print(Y)


# fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion='entropy', )
clf = clf.fit(X, Y)

# plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes', 'No'], filled=True, rounded=True)
plt.show()
