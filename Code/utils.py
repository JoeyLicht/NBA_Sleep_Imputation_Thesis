import df_cleaning
import pandas as pd
from df_cleaning import DATA_PATH, RAW_PATH, CLEAN_PATH, OTHER_PATH, RESULTS_PATH, PLOTS_PATH, FINAL_PATH


def clean_load(path, csv_name):
  '''
  path: string of path to load folder (ie '../Data/')
  csv_name: string of csv name (ie 'NBA_Data.csv')

  Returns df of the csv that is save in the drive at path + csv_name
  This is a cean load because it cleans the date and time types to datetime objects
  '''
  load_path = path + csv_name
  df = pd.read_csv(load_path)
  df = df_cleaning.clean_df(df)
  return df

def save(df, path, csv_name):
  """
  df: pandas dataframe
  path: string of path to save folder (ie '../Data/')
  csv_name: string of csv name (ie 'NBA_Data.csv')

  Saves df to path + csv_name
  """
  save_path = path + csv_name
  df.to_csv(save_path, index=False)
  print(f'Saved to: {save_path}')

def load(path, csv_name):
  '''
  path: string of path to load folder (ie '../Data/')
  csv_name: string of csv name (ie 'NBA_Data.csv')

  Returns df of the csv that is save in the drive at path + csv_name
  '''
  load_path = path + csv_name
  df = pd.read_csv(load_path)
  return df





