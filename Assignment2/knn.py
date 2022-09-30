# -------------------------------------------------------------------------
# AUTHOR: Yi Huang
# FILENAME: knn.py
# SPECIFICATION: Compute LOO-CV error rate for 1NN
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 Hour
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []
error = 0
total = 0

# reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)

# loop your data to allow each instance to be your test set
for i, instance in enumerate(db):
    # add the training features to the 2D array X and remove the instance that will be used for testing in this iteration.
    # For instance, X = [[1, 3], [2, 1,], ...]]. Convert values to float to avoid warning messages

    # transform the original training classes to numbers and add them to the vector Y. Do not forget to remove the instance that will be used for testing in this iteration.
    # For instance, Y = [1, 2, ,...]. Convert values to float to avoid warning messages

    # --> add your Python code here
    X = []
    for unit in db:
        if unit != instance:
            X.append([float(unit[0]), float(unit[1])])

    Y = []
    for unit in db:
        if unit != instance:
            if unit[2] == "+":
                Y.append(1)
            elif unit[2] == "-":
                Y.append(2)

    # fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    # use your test sample in this iteration to make the class prediction. For instance:
    # class_predicted = clf.predict([[1, 2]])[0]
    # --> add your Python code here
    class_predicted = clf.predict([[float(instance[0]), float(instance[1])]])[0]

    # compare the prediction with the true label of the test instance to start calculating the error rate.
    # --> add your Python code here
    if instance[2] == "+":
        true_label = 1
    elif instance[2] == "-":
        true_label = 2

    if class_predicted != true_label:
        error += 1
    total += 1

# print the error rate
# --> add your Python code here
print("LOO-CV error rate:", error/total)
