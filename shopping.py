import csv
import sys
from itertools import islice
from time import strptime


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []

    with open("shopping.csv") as f:
        reader = csv.reader(f)

        line = []

        for row in islice(reader, 1, None):
            line = row[:-1]
            labels.append(1 if row[-1] == "TRUE" else 0)

            for i in range(len(line)):
                if i == 0 or i == 2 or i == 4 or (i >= 11 and i <= 14):
                    line[i] = int(line[i])
                elif i == 1 or i == 3 or i == 5 or (i >= 6 and i <= 9):
                    line[i] = float(line[i])
                elif i == 10:
                    if line[i] == "June":
                        line[i] = 5 #because strip time only takes 3 letter arguments for months
                    else:
                        line[i] = (strptime(line[i], "%b").tm_mon) - 1
                elif i == 15:
                    if line[i] == 'Returning_Visitor':
                        line[i] = 1
                    else:
                        line[i] = 0
                elif i == 16:
                    if line[i] == 'True':
                        line[i] = 1
                    else:
                        line[i] = 0
            evidence.append(line)

    return (evidence, labels)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    return model.fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    positive = 0
    negative = 0
    purchase = 0
    no_purchase = 0

    for i in range(len(labels)):
        if labels[i] == 1:
            positive += 1
            if predictions[i] == 1:
                purchase += 1
        if labels[i] == 0:
            negative += 1
            if predictions[i] == 0:
                no_purchase += 1

    sensitivity = float(purchase / positive)
    specificity = float(no_purchase / negative)

    return(sensitivity, specificity)

if __name__ == "__main__":
    main()
