# -------------------------------------------------------------------------
# AUTHOR: Yi Huang
# FILENAME: decision_tree_2.py
# SPECIFICATION: Train, test, and output the performance of the 3 models created by using each training set on the test set provided
# FOR: CS 4210- Assignment #2
# TIME SPENT: 3 hours 30 minutes
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

Age = {"Young": 1, "Presbyopic": 2, "Prepresbyopic": 3}
Spectacle = {"Myope": 1, "Hypermetrope": 2}
Astigmatism = {"Yes": 1, "No": 2}
Tear_Production_Rate = {"Normal": 1, "Reduced": 2}
categorical_training_classes = {"Yes": 0, "No": 1}

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    # reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(row)

    # transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    # --> add your Python code here
    for instance in dbTraining:
        temp = [Age[instance[0]], Spectacle[instance[1]], Astigmatism[instance[2]], Tear_Production_Rate[instance[3]]]
        X.append(temp)

    # transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    # --> addd your Python code here
    for instance in dbTraining:
        Y.append(categorical_training_classes[instance[4]])

    # loop your training and test tasks 10 times here
    accuracy_ary = []
    for i in range(10):

        # fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
        clf = clf.fit(X, Y)

        # read the test data and add this data to dbTest
        # --> add your Python code here
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for j, row in enumerate(reader):
                if j > 0:  # skipping the header
                    dbTest.append(row)

        confusion_matrix = [[0]*2 for i in range(2)]  # Initialization, used to record the classification
        # [TP,FN]
        # [FP,TN]

        for data in dbTest:
            # transform the features of the test instances to numbers following the same strategy done during training,
            # and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            # where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            # --> add your Python code here
            # predicted class
            temp = [Age[data[0]], Spectacle[data[1]], Astigmatism[data[2]], Tear_Production_Rate[data[3]]]
            class_predicted = clf.predict([temp])[0]

            # compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            # --> add your Python code here
            # actual class
            if data[4] == "Yes":
                true_label = 0
            elif data[4] == "No":
                true_label = 1
            confusion_matrix[true_label][class_predicted] += 1

        # find the lowest accuracy of this model during the 10 runs (training and test set)
        # --> add your Python code here
        accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / (confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1])
        accuracy_ary.append(accuracy)
    accuracy_ary.sort()

    # print the lowest accuracy of this model during the 10 runs (training and test set).
    # your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    # --> add your Python code here
    print("final accuracy when training on", ds, ":", accuracy_ary[0])
