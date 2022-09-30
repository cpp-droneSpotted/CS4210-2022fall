# -------------------------------------------------------------------------
# AUTHOR: Yi Huang
# FILENAME: naive_bayes.py
# SPECIFICATION: Output the classification base on Naïve Bayes strategy
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hours
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# importing some Python libraries
import csv
from sklearn.naive_bayes import GaussianNB

Outlook = {"Sunny": 1, "Overcast": 2, "Rain": 3}
Temperature = {"Hot": 1, "Mild": 2, "Cool": 3}
Humidity = {"High": 1, "Normal": 2}
Wind = {"Weak": 1, "Strong": 2}
PlayTennis = {"Yes": 1, "No": 2}
PlayTennisConvert = {1: "Yes", 2: "No"}

# reading the training data in a csv file
# --> add your Python code here
dbTraining = []
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            dbTraining.append(row)

# transform the original training features to numbers and add them to the 4D array X.
# For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
# --> add your Python code here
X = []
for instance in dbTraining:
    temp = [Outlook[instance[1]], Temperature[instance[2]], Humidity[instance[3]], Wind[instance[4]]]
    X.append(temp)

# transform the original training classes to numbers and add them to the vector Y.
# For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
# --> add your Python code here
Y = []
for instance in dbTraining:
    Y.append(PlayTennis[instance[5]])

# fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

# reading the test data in a csv file
# --> add your Python code here
dbTest = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for j, row in enumerate(reader):
        if j > 0:  # skipping the header
            dbTest.append(row)

# printing the header os the solution
print("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

# use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
# --> add your Python code here
for instance in dbTest:
    Z = [Outlook[instance[1]], Temperature[instance[2]], Humidity[instance[3]], Wind[instance[4]]]

    classification_confidence = clf.predict_proba([Z])[0]
    class_predicted = clf.predict([Z])[0]

    if classification_confidence[class_predicted - 1] >= 0.75:
        print(instance[0].ljust(15) + instance[1].ljust(15) + instance[2].ljust(15) + instance[3].ljust(15) + instance[4].ljust(15) + PlayTennisConvert[class_predicted].ljust(15) + str(classification_confidence[class_predicted - 1]))
