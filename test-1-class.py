from python-toolbox import *


def test_DataSet():
    '''
    Test the DataSet class
    '''
    filename = 'data.csv'
    dataset = DataSet(filename)

    # Test __readFromCSV method
    if dataset._DataSet__readFromCSV(filename):
        print("Super_class DataSet: __readFromCSV method test: PASSED")
    else:
        print("Super_class DataSet: __readFromCSV method test: FAILED")

    # Test __load method
    if dataset._DataSet__load(filename):
        print("Super_class DataSet: __load method test: PASSED")
    else:
        print("Super_class DataSet: __load method test: FAILED")

    # Test clean method
    dataset.clean()
    print("Super_class DataSet: clean method test: PASSED")

    # Test explore method
    if dataset.explore():
        print("Super_class DataSet: explore method test: PASSED")
    else:
        print("Super_class DataSet: explore method test: FAILED")



def test_ClassifierAlgorithm():
    '''
    Test the ClassifierAlgorithm class
    '''
    classifier = ClassifierAlgorithm()
    # Test train method
    if classifier.train():
        print("Super_class ClassifierAlgorithm: train method test: PASSED")
    else:
        print("Super_class ClassifierAlgorithm: train method test: FAILED")

    # Test test method
    if classifier.test():
        print("Super_class ClassifierAlgorithm: test method test: PASSED")
    else:
        print("Super_class ClassifierAlgorithm: test method test: FAILED")



def test_DataSet_subclasses():
    '''
    Test the subclasses of DataSet class
    '''
    time_series_dataset = TimeSeriesDataSet('time_series.csv')
    text_dataset = TextDataSet('text.csv')
    quant_dataset = QuantDataSet('quant.csv')
    qual_dataset = QualDataSet('qual.csv')

    if isinstance(time_series_dataset, DataSet):
        print("TimeSeriesDataSet is a subclass of DataSet: PASSED")
    if isinstance(text_dataset, DataSet):
        print("TextDataSet is a subclass of DataSet: PASSED")
    if isinstance(quant_dataset, DataSet):
        print("QuantDataSet is a subclass of DataSet: PASSED")
    if isinstance(qual_dataset, DataSet):
        print("QualDataSet is a subclass of DataSet: PASSED")
    if time_series_dataset.filename == 'time_series.csv' and \
        text_dataset.filename == 'text.csv'  and \
        quant_dataset.filename == 'quant.csv'   and \
        qual_dataset.filename == 'qual.csv':    
        print("DataSet subclasses test: PASSED")
    else:
        print("DataSet subclasses test: FAILED")

def test_ClassifierAlgorithm_classes():
    '''
    Test the subclasses of ClassifierAlgorithm class
    '''

    simple_knn_classifier = simpleKNNClassifier()
    kdtree_knn_classifier = kdTreeKNNClassifier()

    if isinstance(simpleKNNClassifier(), ClassifierAlgorithm):
        print("simpleKNNClassifier is a subclass of ClassifierAlgorithm: PASSED")
    if isinstance(kdTreeKNNClassifier(), ClassifierAlgorithm):
        print("kdTreeKNNClassifier is a subclass of ClassifierAlgorithm: PASSED")

    if simple_knn_classifier is not None and \
        kdtree_knn_classifier is not None:
        print("ClassifierAlgorithm: subclasses test: PASSED")
    else:
        print("ClassifierAlgorithm: subclasses test: FAILED")


def test_Experiment():
    '''
    Test the Experiment class
    '''

    experiment = Experiment()
    # Test runCrossVal method
    if experiment.runCrossVal(5):
        print("Experiment: runCrossVal method test: PASSED")
    else:
        print("Experiment: runCrossVal method test: FAILED")

    # Test score method
    if experiment.score():
        print("Experiment: score method test: PASSED")
    else:
        print("Experiment: score method test: FAILED")

    # Test __confusionMatrix method
    if experiment._Experiment__confusionMatrix():
        print("Experiment: __confusionMatrix method test: PASSED")
    else:
        print("Experiment: __confusionMatrix method test: FAILED")

    if experiment is not None:
        print("Experiment class test: PASSED")
    else:
        print("Experiment class test: FAILED")


if __name__ == '__main__':
    '''
    Test all the classes and methods
    '''
    print('Test for class DataSet:')
    test_DataSet()
    print('\n')
    print('Test for class ClassifierAlgorithm:')
    test_ClassifierAlgorithm()
    print('\n')
    print('Test for subclasses of class DataSet:')
    test_DataSet_subclasses()
    print('\n')
    print('Test for subclasses of class ClassifierAlgorithm:')
    test_ClassifierAlgorithm_classes()
    print('\n')
    print('Test for class Experiment:')
    test_Experiment()