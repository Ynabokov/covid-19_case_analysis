import pandas as pd
import numpy as np
import math
from helper_1_4 import transfrom_locations
from helper_1_5 import join_cases_locations

DATA_DIR = '../data/'
LOCATION_FILENAME = DATA_DIR + 'location.csv'
CASES_TRAIN_FILENAME = DATA_DIR + 'cases_train.csv'
CASES_TEST_FILENAME = DATA_DIR + 'cases_test.csv'


def clean(data):
    for i in range(len(data)):

# Clean age
        value = (data.iloc[i]).at["age"]
        if (pd.notna(value)):
            # Values like '10.0'
            if ('.' in value):
                value, drop = value.split('.')
                data.at[i,"age"] = value
            # Values like '5-20', '-80', '20-'
            elif ('-' in value):
                lower, upper = value.split('-')
                if (lower == ''):
                    lower = 0
                else:
                    lower = int(lower)
                if (upper == ''):
                    upper = 100
                elif (any(c.isalpha() for c in upper)):
                    data.at[i,"age"] = pd.NA
                    continue
                else:
                    upper = int(upper)
                value = str(int((upper+lower)/2))
                data.at[i,"age"] = value
            # Values like '80+'
            elif ('+' in value):
                lower, upper = value.split('+')
                lower = int(lower)
                upper = 100
                value = str(int((upper+lower)/2))
                data.at[i,"age"] = value
            # Values like '5 month'
            elif ('month' in value):
                lower, drop = value.split(' month')
                lower = int(int(lower)/12)
                value = str(lower)
                data.at[i,"age"] = value    

# Clean date_confirmation
        value = (data.iloc[i]).at["date_confirmation"]
        if (pd.notna(value)):
        # Values like 01.01.2020 - 02.02.2020 and 01.01.2020-02.02.2020
            if ('-' in value):
                value, drop = value.split('-')
                if (' ' in value):
                    value, drop = value.split(' ')
                data.at[i,"date_confirmation"] = value

    return data
#end clean

def impute(data):
    agesum = 0
    ageentries = 0
    for i in range(len(data)):
        value = (data.iloc[i]).at["age"]
        if (pd.notna(value) and value != ''):
            agesum += int(value)
            ageentries += 1

    average = int(agesum/ageentries)

    for i in range(len(data)):
        value = (data.iloc[i]).at["age"]
        if (pd.isna(value) or value == ''):
            data.at[i,"age"] = average

        value = (data.iloc[i]).at["sex"]
        if (pd.isna(value) or value == ''):
            data.at[i,"sex"] = 'unknown'

        value = (data.iloc[i]).at["country"]
        if (pd.isna(value) or value == ''):
            data.at[i,"country"] = (data.iloc[i]).at["province"]

    return data
#end impute

def remove_outliers (data):
    data = data[data['country'].notna()]
    data = data[data['latitude'].notna()]
    data = data[data['longitude'].notna()]
    return data  


def main():
    ''' Start of the program '''

    # Import

    train_dataset = pd.read_csv(CASES_TRAIN_FILENAME)
    test_dataset = pd.read_csv(CASES_TEST_FILENAME)
    location_dataset = pd.read_csv(LOCATION_FILENAME)

    '''1.2'''
    '''1.3'''

    # Cleaning

    train_dataset = clean(train_dataset)
    train_dataset = impute(train_dataset)
    train_dataset = remove_outliers(train_dataset)
    test_dataset = clean(test_dataset)
    test_dataset = impute(test_dataset)
    test_dataset = remove_outliers(test_dataset)

    '''1.4'''

    location = pd.read_csv(LOCATION_FILENAME)
    location_transformed = transfrom_locations(location)

    '''1.5'''
    join_cases_locations(train_dataset, location_transformed, 'cases_train_processed.csv')
    join_cases_locations(test_dataset, location_transformed, 'cases_test_processed.csv')


if __name__ == '__main__':
    main()
