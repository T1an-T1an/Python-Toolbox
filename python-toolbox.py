#* <JingdaProj3>.<py>
#*
#* ANLY 555 <Spring 2023>
#* Project <3>
#*
#* Due on: <Mar, 23, 2023>
#* Author(s): <Jingda Yang>
#*
#*
#* In accordance with the class policies and Georgetown's
#* Honor Code, I certify that, with the exception of the
#* class resources and those items noted below, I have neither
#* given nor received any assistance on this project other than
#* the TAs, professor, textbook and teammates.
#*


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud


class DataSet: # base class
    def __init__(self, filename):
        '''
        Initialize the object's state
        '''
        self.filename = filename
        self._data = None
        self._type = None
        self.__readFromCSV(filename)
        self.__load()
    def __readFromCSV(self, filename): # private method to read csv file
        '''
        Use this method to read csv file.
        '''
        import pandas as pd
        try:
            self._data = pd.read_csv(filename)
        except FileNotFoundError:
            print(f"Error: File {filename} not found.")
            raise

    def __load(self): # private method to load data
        '''
        Load dataset from file
        '''
        self.filename = input("Enter the name of the file: ")
        self._type = input("Enter the type of data set (e.g. Quantitative, Qualitative, TimeSeries, Text): ")
        print(f"Loading {self._type} data set from file: {self.filename}")
        self.__readFromCSV(self.filename)

        
    def clean(self): # public method to clean data
        '''
        Clean dataset
        '''
        print('Dataset is cleaned.')

    def explore(self): # public method to explore data
        print('Dataset is explored.')



class QuantDataSet(DataSet):
    '''
    Inherit from DataSet class
    '''
    def __init__(self, filename):
        super().__init__(filename)
        if self._type == "Quantitative":
           self._DataSet__readFromCSV(self.filename)
        else:
            raise ValueError("Invalid dataset type.")
            
    
    def clean(self):
        '''
        Over write the clean method
        '''
        means = self._data.mean()
        self._data = self._data.fillna(means)
        print('Dataset is cleaned.')

    def explore(self): 
        '''
        Over write the explore method
        '''
        sales_by_week = self._data.sum(axis=1)
        plt.figure(figsize=(12, 6))
        plt.plot(sales_by_week, color='blue')
        plt.xlabel('Week')
        plt.ylabel('Total Sales')
        plt.title('Weekly Sales Transactions')
        plt.show()

        plt.figure(figsize=(12,12))
        self._data[self._data["Product_Code"] == "P1"].set_index("Product_Code").T.plot()
        plt.title("Sales Trend for Product P1")
        plt.xlabel("Week")
        plt.ylabel("Sales")
        plt.xticks(rotation='vertical')
        plt.show()
        print('Dataset is explored.')



class QualDataSet(DataSet):
    '''
    Inherit from DataSet class
    '''
    def __init__(self, filename):
        super().__init__(filename)
        if self._type == "Qualitative":
            self._DataSet__readFromCSV(self.filename)
        else:
            raise ValueError("Invalid dataset type.")
    
    def clean(self):
        '''
        Over write the clean method
        '''
        modes = self._data.mode()
        self._data = self._data.fillna(modes)
        print('Dataset is cleaned.')

    def explore(self):
        '''
        Over write the explore method
        '''
        data = self._data.drop(0)
        q2_data = data[['Q2']]
        
        plt.hist(q2_data.values)

        # set the plot title and axis labels
        plt.title("What is your gender? - Selected Choice")
        plt.xlabel("Gender")
        plt.ylabel("Frequency")

        # show the plot
        plt.show()

        q3_data = data[['Q1']]
        
        plt.hist(q3_data.values)  # plot only top 10 countries
        
        # set the plot title and axis labels
        plt.title("What is your age (# years)?")
        plt.xlabel("Age Range")
        plt.ylabel("Frequency") 
        
        # rotate x-axis labels
        plt.xticks(rotation='vertical')

        # show the plot
        plt.show()
        print('Dataset is explored.')


class TimeSeriesDataSet(DataSet):
    '''
    Inherit from DataSet class
    '''
    def __init__(self, filename):
        super().__init__(filename)
        if self._type == "TimeSeries":
            self._DataSet__readFromCSV(self.filename)
        else:
            raise ValueError("Invalid dataset type.")
    
    def clean(self):
        '''
        Over write the clean method
        fill missing values with the median of the column
        '''
        window_size = 3
        valid_cols = self._data.columns[1:]
        self._data[valid_cols] = self._data[valid_cols].rolling(window_size).median()
        self._data[valid_cols] = self._data[valid_cols].fillna(method='ffill')
        print('Dataset is cleaned.')
        return self._data

    
    def explore(self):
        '''
        Over write the explore method
        '''
        # plot the time series data
        self._data['Date'] = pd.to_datetime(self._data['Date'])
        plt.figure(figsize=(12,6))
        plt.plot(self._data['Date'], self._data['Adj Close'])
        plt.title('TSLA Stock Adj Close Price')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.show()

        plt.figure(figsize=(12,6))
        plt.plot(self._data['Date'], self._data['Close'])
        plt.title('TSLA Stock Close Price')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.show()
        print('Dataset is explored.')



class TextDataSet(DataSet):
    '''
    Inherit from DataSet class
    '''
    def __init__(self, filename):
        super().__init__(filename)
        if self._type == "Text":
            self._DataSet__readFromCSV(self.filename)
        else:
            raise ValueError("Invalid dataset type.")
    
    def clean(self):
        '''
        Over write the clean method
        '''
        # select only column E
        e_col = self._data['text']
        
        # remove stop words from each row of column E
        stop_words = set(stopwords.words('english'))
        e_col = e_col.apply(lambda x: ' '.join([word.lower() for word in x.split() if word.lower() not in stop_words]))
        
        # update the cleaned data in the original DataFrame
        self._data['text'] = e_col
        print('Dataset is cleaned.')
        return self._data

    
    def explore(self):
        '''
        Over write the explore method
        '''
        # check if column contains only string or text data
        if not self._data['text'].apply(lambda x: isinstance(x, str)).all():
            raise ValueError("Column contains non-string data.")
        
        wordcloud = WordCloud().generate(' '.join(self._data['text']))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title('World Cloud for Yelp Data', fontsize = 16)
        plt.show()

        plt.figure(figsize=(12,5))
        self._data['stars'].value_counts().plot(kind='bar',alpha=0.5,color='b',label='ratings')
        plt.legend()
        plt.title('Yelp Stars', fontsize = 16)
        plt.xlabel('Rate', fontsize = 12)
        plt.ylabel('Frequency',fontsize = 12)
        plt.show()
        print('Dataset is explored.')






class ClassifierAlgorithm:
    def __init__(self):
        self.trainingData = None
        self.trueLabels = None
        self.predictedLabels = None

    def train(self, trainingData, trueLabels):
        self.trainingData = trainingData
        self.trueLabels = trueLabels
        return True

    def test(self, testData):
        return True


class Experiment:
    def __init__(self, data, labels, classifiers):
        self.data = data
        self.labels = labels
        self.classifiers = classifiers
        

    # define runCrossVal method
    def runCrossVal(self, kFolds):
        '''
        Run k-fold cross-validation experiment
        '''
        print("Experiment runCrossVal method invoked")

        numSamples = self.data.shape[0] #number of samples
        numClassifiers = len(self.classifiers)  #number of classifiers
        predictedLabelsMatrix = np.zeros((numSamples, numClassifiers), dtype=int)   #matrix to store predicted labels

        numk = self.classifiers[0].k    #set value of k
        foldSize = numSamples // kFolds #size of each fold, only integer needed

        for fold in range(kFolds):
            #for each fold, get the test and train indices
            test_indices = np.arange(fold * foldSize, (fold + 1) * foldSize)
            
            train_indices = np.arange(0, fold * foldSize)
            train_indices = np.append(train_indices, np.arange((fold + 1) * foldSize, numSamples))


            # get the test and train data
            trainData = self.data[train_indices]
            trainLabels = self.labels[train_indices]
            testData = self.data[test_indices]
            testLabels = self.labels[test_indices]


            
            #train and test each classifier
            '''
            If a classifier is not appropriate for the data set, 
            the runCrossVal method will automatically throw any inherent exceptions 
            that the classifiers themselves have provided within their train and test methods.
            '''
            for i in range(numClassifiers):
                classifier = self.classifiers[i]
                classifier.train(trainData, trainLabels)
                predictedLabels = classifier.test(testData, numk)
                predictedLabelsMatrix[test_indices, i] = predictedLabels

        self.predictedLabelsMatrix = predictedLabelsMatrix
        return predictedLabelsMatrix


    # define score method
    def score(self):
        '''
        Score results of the experiment
        '''
        print("Experiment score method invoked")
        accuracies = []

        #calculate accuracy for each classifier if we have multiple classifiers
        for i in range(len(self.classifiers)):
            correct_predictions = np.sum(self.labels == self.predictedLabelsMatrix[:, i])
            num_labels = len(self.labels)
            accuracy = correct_predictions / num_labels
            accuracies.append(accuracy)

        #print accuracies
        print("Classifier Accuracies:")

        for i in range(len(accuracies)):
            accu = accuracies[i]
            print("Classifier {}: {:.4f}".format(i+1, accu))


    # define confusionMatrix method
    def confusionMatrix(self):
        '''
        Generate confusion matrix for experiment
        '''
        print("Experiment confusionMatrix method invoked")

        #calculate confusion matrix for each classifier if we have multiple classifiers
        numClassifiers = len(self.classifiers)
        numClasses = len(np.unique(self.labels))

        for i in range(numClassifiers):
            #initialize confusion matrix so we can store the values in it
            cnfsn_matrix = np.zeros((numClasses, numClasses), dtype=int)

            predictedLabels = self.predictedLabelsMatrix[:, i]
            for j in range(len(self.labels)):
                true_label = self.labels[j]
                predicted_label = predictedLabels[j]
                cnfsn_matrix[true_label][predicted_label] = cnfsn_matrix[true_label][predicted_label] + 1
                
            #print confusion matrix
            print(f"Confusion Matrix for Classifier {i+1}:")
            print(cnfsn_matrix)
            print("\n")




class simpleKNNClassifier(ClassifierAlgorithm):
    '''
    simpleKNNClassifier class inherits from ClassifierAlgorithm class
    '''
    def __init__(self):
        super().__init__()
        # self.k = k
        # self.trainingData = None
        # self.trueLabels = None
        # self.predictedLabels = None

    # train and test methods are inherited from ClassifierAlgorithm class
    def train(self, trainingData, trueLabels):
        self.trainingData = trainingData
        self.trueLabels = trueLabels
        return True

    def test(self, testData, k):
        self.k = k
        numTestSamples = testData.shape[0]
        predictedLabels = np.zeros(numTestSamples, dtype=int)

        #calculate euclidean distance, sort and get the k closest labels
        for i in range(numTestSamples):
            distances = np.sqrt(np.sum(np.square(self.trainingData - testData[i]), axis=1))
            closest_k_indices = np.argsort(distances)[:k]
            closest_k_labels = self.trueLabels[closest_k_indices]
            predictedLabels[i] = np.bincount(closest_k_labels).argmax() #get the most common label

        self.predictedLabels = predictedLabels #save the predicted labels
        return predictedLabels
    


class kdTreeKNNClassifier(ClassifierAlgorithm):
    '''
    kdTreeKNNClassifier class inherits from ClassifierAlgorithm class
    '''
    def __init__(self):
        print("kdTreeKNNClassifier __init__ method invoked")
        super().__init__()