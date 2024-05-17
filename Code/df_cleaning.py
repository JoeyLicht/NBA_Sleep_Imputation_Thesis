import pandas as pd
import utils
from datetime import datetime
import numpy as np
import os
from IPython.display import display

# Define relative paths
DATA_PATH = "./Data/"
RAW_PATH = f"{DATA_PATH}Raw_Data/"
CLEAN_PATH = f"{DATA_PATH}Cleaned/"
OTHER_PATH = f"{DATA_PATH}Other/"
RESULTS_PATH = "./Results/"
PLOTS_PATH = f"{RESULTS_PATH}Plots/"
FINAL_PATH = "./Final Thesis Figures/"

def clean_date_and_time(df):
    """
    Converts specific date and time columns in the dataframe to datetime objects.

    Parameters:
    df (pandas.DataFrame): The input dataframe.

    Returns:
    pandas.DataFrame: The dataframe with converted date/time columns.
    """
    date_columns = ['Date', 'Date_First_Logged']
    time_columns = ['Bedtime Start', 'Bedtime End']

    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col], format='%m/%d/%Y')
        except:
            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')

    for col in time_columns:
        df[col] = pd.to_datetime(df[col], format='%I:%M %p').dt.strftime('%H:%M')
        df[col] = pd.to_datetime(df['Date'].dt.date.astype(str) + ' ' + df[col])
        if col == 'Bedtime End':
            df.loc[df['Bedtime End'] < df['Bedtime Start'], 'Bedtime End'] += pd.DateOffset(days=1)

    columns_to_convert = ['Duration', 'Awake', 'Light', 'Onset Latency']
    for column in columns_to_convert:
        df[column] = df[column].apply(lambda x: datetime.strptime(x, '%H:%M:%S').time() if pd.notnull(x) else np.nan)

    return df

def calculate_logged_dates(df):
    """
    Adds 'Date_First_Logged' and 'Days_Since_First_Logged' columns to the dataframe.

    Parameters:
    df (pandas.DataFrame): The input dataframe.

    Returns:
    pandas.DataFrame: The dataframe with the new columns.
    """
    df['Date_First_Logged'] = df.groupby('ID')['Date'].transform('min')
    df['Days_Since_First_Logged'] = (df['Date'] - df['Date_First_Logged']).dt.days
    return df

def clean_df(df):
    """
    Cleans the dataframe by performing various operations including date conversion,
    removing unnecessary columns, and calculating additional columns.

    Parameters:
    df (pandas.DataFrame): The input dataframe.

    Returns:
    pandas.DataFrame: The cleaned dataframe.
    """
    df = clean_date_and_time(df)
    df = remove_unnecessary_columns(df)
    df = df.drop_duplicates(subset=['ID', 'Date'])
    df = df.sort_values(by=['ID', 'Date'])
    df['Total Nights'] = df.groupby('ID')['ID'].transform('count')
    df['Running Total Nights'] = df.groupby('ID')['ID'].cumcount() + 1
    df['Date_First_Logged'] = df.groupby('ID')['Date'].transform('min')
    df['Days_Since_First_Logged'] = (df['Date'] - df['Date_First_Logged']).dt.days
    return df

def remove_unnecessary_columns(df):
    """
    Removes unnecessary columns from the dataframe based on a utility file.

    Parameters:
    df (pandas.DataFrame): The input dataframe.

    Returns:
    pandas.DataFrame: The dataframe with unnecessary columns removed.
    """
    column_utility = utils.load(OTHER_PATH, 'NBA_Data_Column_Utility_NEW.csv')
    columns_to_drop = column_utility[column_utility['Useful/Accurate'] == 'No']['Column Name'].values
    return df.drop(columns=columns_to_drop)

def merge_sleep_df():
    """
    Merges original and new sleep dataframes and saves the combined dataframe.
    """
    original_df = utils.load(RAW_PATH, 'sleep_original_period.csv')
    new_df = utils.load(RAW_PATH, 'sleep_new_period.csv')

    print('Columns in original df but not in new df:')
    print(original_df.columns.difference(new_df.columns))

    print('Columns in new df but not in original df:')
    print(new_df.columns.difference(original_df.columns))

    print('Columns in both dataframes:')
    print(new_df.columns.intersection(original_df.columns))

    original_df = original_df.drop(columns=['ID']).rename(columns={'About': 'ID'})
    original_df = original_df.drop(columns=['Last_X_Nights', 'Total Nights'])
    new_df = new_df.drop(columns=new_df.columns.difference(original_df.columns))

    assert all(original_df.columns == new_df.columns)

    combined_df = pd.concat([original_df, new_df])
    utils.save(combined_df, CLEAN_PATH, 'sleep_all_period_raw.csv')

def column_description(df, column_name):
    """
    Provides a description of a column including basic stats, data types, and unique values.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    column_name (str): The column to describe.

    Returns:
    tuple: Description, types of values, and sorted unique values of the column.
    """
    def column_type(df, column_name):
        return df[column_name].apply(type).unique()

    def sorted_unique_values(df, column_name):
        return df[column_name].sort_values().unique()
    
    return (df[column_name].describe(), column_type(df, column_name), sorted_unique_values(df, column_name))

def column_description_df(df, csv_name):
    """
    Creates a dataframe describing each column and saves it as a CSV file.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    csv_name (str): The name of the CSV file to save the description.
    """
    rows = []
    for column_name in df.columns:
        description, col_type, unique_values = column_description(df, column_name)
        rows.append({'Column Name': column_name, 'Description': description, 'Type': col_type, 'Unique Values': unique_values})

    result_df = pd.DataFrame(rows)
    utils.save(result_df, OTHER_PATH, csv_name)

def convert_to_milliseconds(x):
    """
    Converts a datetime object to milliseconds.

    Parameters:
    x (datetime): The datetime object to convert.

    Returns:
    int: The total milliseconds.
    """
    return x.hour * 3600 * 1000 + x.minute * 60 * 1000 + x.second * 1000

def convert_to_hours(x):
    """
    Converts a datetime object to hours.

    Parameters:
    x (datetime): The datetime object to convert.

    Returns:
    float: The total hours, rounded to 2 decimal places.
    """
    return round(x.hour + x.minute / 60 + x.second / 3600, 2)

if __name__ == '__main__':
    df = utils.clean_load(CLEAN_PATH, 'sleep_all_period_raw.csv')
    display(df.head())
    utils.save(df, CLEAN_PATH, 'sleep_all_period_clean.csv')
