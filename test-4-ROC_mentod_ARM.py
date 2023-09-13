from python-toolbox import simpleKNNClassifier, Experiment, TransactionDataSet, Rule, DataSet

'''
Test part a for this project.


I have used the Iris dataset to test simpleKNNClassifier at first, 
and use the result to test function ROC in class Experiment. 
I prepare my data before I use it to train and test the classifier.
First I use random package to shuffle the data and split it into training and testing sets with ratio 0.7(training) and 0.3(testing). 
Then I use the training data and labels to train the classifier and use the test data to test the classifier. 
In class experiment, method test() will return a numpy array of predicted labels.
Because I prepare the data in advance, I did not put codes about preparing data in class Experiment method test().
Finally I calculate the accuracy of the predictions and test class Experiment, then plot ROC curves. 
Thank you so much!
'''

def main():
    import pandas as pd
    import numpy as np
    import random
    import warnings
    warnings.filterwarnings('ignore')
    '''
    Prepare data.
    '''

    csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    # using the attribute information as the column names
    col_names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Class']
    iris =  pd.read_csv(csv_url, names = col_names)
    # define a label encoding dictionary
    label_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

    # convert object labels to integer labels using the label encoding dictionary
    features = iris.iloc[:, :-1].values
    labels = iris['Class'].map(label_dict).values

    # randomly shuffle the data
    data = list(zip(features, labels))
    random.shuffle(data)

    # split data into training and testing sets with ratio 0.7(training) and 0.3(testing)
    split_ratio = 0.7
    split_index = int(split_ratio * len(data))

    trainingData, trueLabels = zip(*data[:split_index])
    testData, testLabels = zip(*data[split_index:])

    # convert to numpy arrays
    trainingData = np.array(trainingData)
    trueLabels = np.array(trueLabels)
    testData = np.array(testData)
    testLabels = np.array(testLabels)

    '''
    Test simpleKNNClassifier at first.
    '''
    # create a simpleKNNClassifier object
    knn_classifier = simpleKNNClassifier()

    # train the classifier using the training data and labels
    knn_classifier.train(trainingData, trueLabels)

    # test the classifier using the test data and choose a value for k
    k = 5
    predictedLabels = knn_classifier.test(testData, k = 5)
    print(predictedLabels)

    # calculate the accuracy of the predictions. I also use this in class experiment to test the accuracy of the predictions.
    numCorrectPredictions = np.sum(predictedLabels == testLabels)
    accuracy = numCorrectPredictions / len(testLabels)
    print(f"Accuracy of simpleKNNClassifier with k={k}: {accuracy:.4f}")

    '''
    Test Experiment class.
    '''

    # Create two instances of simpleKNNClassifier with different k values
    knn_classifier1 = simpleKNNClassifier()
    knn_classifier1.k = 5

    knn_classifier2 = simpleKNNClassifier()
    knn_classifier2.k = 5

    # Create an instance of the Experiment class
    experiment = Experiment(testData, predictedLabels, [knn_classifier1])
    experiment_2 = Experiment(testData, predictedLabels, [knn_classifier2])

    # Run k-fold cross-validation
    predictedLabelsMatrix = experiment.runCrossVal(kFolds=5)
    predictedLabelsMatrix_2 = experiment_2.runCrossVal(kFolds=3)

    '''
    predictedLabelsMatrix is stored in a list of numpy arrays. 
    you can print it out to see the results if needed. Thanks!
    '''
    #print(predictedLabelsMatrix)
    #print(predictedLabelsMatrix_2)

    # Score the results
    experiment.score()
    experiment_2.score()

    # Generate confusion matrices
    experiment.confusionMatrix()
    experiment_2.confusionMatrix()

    # Generate ROC curves
    experiment.ROC()
    experiment_2.ROC()


    '''
    Test part c for this project.

    I have used the transaction dataset to test TransactionDataSet class.
    I create a ramdom dataset with 100000 rows and 7 columns, and save it as a csv file.
    Then I use TransactionDataSet class to read the csv file and explore the dataset.
    Because the dataset I created is clean, so I did not use any data cleaning methods.
    '''

    # create a random dataset with 100000 rows and 7 columns
    np.random.seed(123)
    arr = np.random.randint(2, size=(100000, 7))
    df = pd.DataFrame(arr, columns=['Item1', 'Item2', 'Item3', 'Item4', 'Item5', 'Item6', 'Item7'])

    # save the dataset as a csv file
    df.to_csv('example.csv', index=False)

    # use TransactionDataSet class to read the csv file and explore the dataset
    transaction_data = TransactionDataSet('example.csv')
    transaction_data.explore()
if __name__ == "__main__":
    main()
