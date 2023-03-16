import env
import acquire as acq
import pandas as pd
import matplotlib as plt
import os

def prep_iris():
    '''
    function takes in data from aquire.get_titanic_data(),
    applies preparatory steps to the dataset, then splits
    the dataset into train, validate, and test groups.
    '''
    iris_df = acq.get_iris_data()
    iris_df.drop(['species_id'], axis=1, inplace=True)
    iris_df.rename(columns={'species_name' : 'species'}, inplace=True)
    dummy_df = pd.get_dummies(iris_df.species, dummy_na=False, drop_first=True)
    iris_df = pd.concat([iris_df, dummy_df], axis=1)
    return iris_df


def prep_titanic(df):
    '''
    This function will drop any duplicate observations, 
    drop ['deck', 'embarked', 'class', 'age'], fill missing embark_town with 'Southampton'
    and create dummy vars from sex and embark_town. 
    '''
    df = df.drop_duplicates()
    df = df.drop(columns=['deck', 'embarked', 'class', 'age'])
    df['embark_town'] = df.embark_town.fillna(value='Southampton')
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    return df



def train_validate_test_split(df, target, seed=9751):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test