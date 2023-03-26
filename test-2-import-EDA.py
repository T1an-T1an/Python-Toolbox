from JingdaYangProj2 import QuantDataSet, QualDataSet, TimeSeriesDataSet, TextDataSet
 
'''
After select all code and run, First please input one of them from ["Quant", "Qual", "TimeSeries", "Text"],
then input the file name, 
Sales_Transactions_Dataset_Weekly.csv is Quant
multiple_choice_responses.csv is Qual
tsla_2.csv is TimeSeries
yelp.csv is Text.

For example, 
if user want to test yelp.csv, the user need to input 'Text', then 'yelp.csv', then confirm again 'Text'
if user want to test multiple_choice_responses.csv, the user need to input 'Qual', then 'multiple_choice_responses.csv', then confirm again 'Qualitative'
When you finish, please input ''Done'' to exit.
Ther are also some warning messages when I read these csv file, but they will not influence my code.
Since we are not allowed to use any packages to remove them, so I just left them.

Thank you!!!
'''

if __name__ == '__main__':
    valid_datasets = ["Quant", "Qual", "TimeSeries", "Text"]
    
    # prompt the user to enter the name of the data set they want to test first
    first_dataset = input("Enter the name of the data set you want to test first (Quant, Qual, TimeSeries, Text): ")
    
    # check if the input is valid
    while first_dataset not in valid_datasets:
        print("Invalid input. Please enter one of the following: Quant, Qual, TimeSeries, Text.")
        first_dataset = input("Enter the name of the data set you want to test first (Quant, Qual, TimeSeries, Text): ")

    # create an instance of the chosen data set class and load the data from file
    if first_dataset == "Quant":
        data = QuantDataSet("Sales_Transactions_Dataset_Weekly.csv")
    elif first_dataset == "Qual":
        data = QualDataSet("multiple_choice_responses.csv")
    elif first_dataset == "TimeSeries":
        data = TimeSeriesDataSet("tsla_2.csv")
    elif first_dataset == "Text":
        data = TextDataSet("yelp.csv")

    # clean and explore the data set
    data.clean()
    data.explore()

    # let the user to choose the next data set to test
    next_dataset = input("Enter the name of the next data set you want to test (Quant, Qual, TimeSeries, Text), or 'Done' to exit: ")
    while next_dataset != "Done":
        # check if the input is valid
        while next_dataset not in valid_datasets:
            print("Invalid input. Please enter one of the following: Quant, Qual, TimeSeries, Text.")
            next_dataset = input("Enter the name of the next data set you want to test (Quant, Qual, TimeSeries, Text), or 'Done' to exit: ")
        
        # create an instance of the chosen data set class and load the data from file
        if next_dataset == "Quant":
            data = QuantDataSet("Sales_Transactions_Dataset_Weekly.csv")
        elif next_dataset == "Qual":
            data = QualDataSet("multiple_choice_responses.csv")
        elif next_dataset == "TimeSeries":
            data = TimeSeriesDataSet("tsla_2.csv")
        elif next_dataset == "Text":
            data = TextDataSet("yelp.csv")

        # clean and explore the data set
        data.clean()
        data.explore()

        # prompt the user to choose the next data set to test
        next_dataset = input("Enter the name of the next data set you want to test (Quant, Qual, TimeSeries, Text), or 'Done' to exit: ")
