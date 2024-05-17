from sleep_imputation import *
from df_cleaning import *
from utils import *
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from df_cleaning import DATA_PATH, RAW_PATH, CLEAN_PATH, OTHER_PATH, RESULTS_PATH, PLOTS_PATH, FINAL_PATH

SLEEP_FEATURE_COLUMNS = ['HR Lowest', "HR Average", "rMSSD", "Breath Average", "Duration Integer hr"]
MAX_LAG = 7

def clean_athletic_df(athletic_df):
    """
    Cleans the athletic dataframe by converting date to datetime, 
    dropping rows with null IDs, and converting IDs to integers.
    
    Parameters:
    athletic_df (pandas.DataFrame): The athletic dataframe to be cleaned.

    Returns:
    pandas.DataFrame: The cleaned athletic dataframe.
    """
    athletic_df['Date'] = pd.to_datetime(athletic_df['Date'], format='%m/%d/%Y')
    athletic_df = athletic_df[athletic_df['ID'].notnull()]
    athletic_df['ID'] = athletic_df['ID'].astype(int)
    return athletic_df

def run_regression(df, sleep_feature, athletic_feature, lag, average, plot=False):
    """
    Runs a regression analysis between sleep features and athletic features.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data.
    sleep_feature (str): The sleep feature to be used as an independent variable.
    athletic_feature (str): The athletic feature to be used as a dependent variable.
    lag (int): The number of days prior to the test date. If 0, the test date is used.
    average (bool): If True, the average from nights 0 to lag is used.
    plot (bool): If True, a scatter plot with the regression line is displayed.

    Returns:
    tuple: Slope, R-squared value, and the number of points used in the regression.
    """
    df = df.copy()
    min_entries = 2
    df = df.groupby('ID').filter(lambda x: x[athletic_feature].count() >= min_entries)

    sleep_subset = [f'{sleep_feature} {i}' for i in range(1, lag + 1)] if average else [f'{sleep_feature} {lag}']
    subset = sleep_subset.copy() + [athletic_feature]
    df = df.dropna(subset=subset)

    if average:
        sleep_feature = f'{sleep_feature} {lag + 1} day average'
        df[sleep_feature] = df[sleep_subset].mean(axis=1)

    y = df[athletic_feature]
    X = sm.add_constant(df[sleep_feature])

    model = sm.OLS(y, X)
    results = model.fit()

    if plot:
        plt.scatter(df[sleep_feature], y)
        plt.plot(df[sleep_feature], results.predict(X), color='red')
        plt.xlabel(sleep_feature)
        plt.ylabel(athletic_feature)
        plt.title(f'{sleep_feature} vs {athletic_feature} \n ({len(df)} points, slope: {results.params[sleep_feature]:.2f}, R^2: {results.rsquared:.2f})')
        plt.show()

    return results.params[sleep_feature], results.rsquared, len(df)

def normalize_helper(df, feature_columns):
    """
    Normalizes each feature column for each unique ID by dividing by the mean of the feature column for that ID.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data.
    feature_columns (list of str): List of feature columns to be normalized.

    Returns:
    pandas.DataFrame: The dataframe with normalized columns.
    """
    for feature in feature_columns:
        df[feature] = df[feature] / df.groupby('ID')[feature].transform('mean')
    return df

def regression_wrapper(merged_df, sleep_columns, athletic_columns, plot=False, save_title=None, save=False):
    """
    Wrapper function to run multiple regressions and save the results.

    Parameters:
    merged_df (pandas.DataFrame): The merged dataframe.
    sleep_columns (list of str): List of sleep feature columns.
    athletic_columns (list of str): List of athletic feature columns.
    plot (bool): If True, plots are generated for each regression.
    save_title (str): The title for the saved results file.
    save (bool): If True, the results are saved to a file.
    """
    results_columns = ['R^2', 'Num Points', 'Slope', 'Sleep Feature', 'Athletic Feature', 'Max Night Lag', 'Average']
    results = pd.DataFrame(columns=results_columns)

    for average in [True, False]:
        for lag in range(0, MAX_LAG):
            for sleep_feature in sleep_columns:
                for athletic_feature in athletic_columns:
                    if lag == 0 and average:
                        continue
                    slope, r_squared, points = run_regression(merged_df, sleep_feature, athletic_feature, lag, average, plot=plot)
                    new_row = pd.DataFrame([[r_squared, points, slope, sleep_feature, athletic_feature, lag + 1, average]], columns=results_columns)
                    results = pd.concat([results, new_row], ignore_index=True)

    results = results.sort_values(by='R^2', ascending=False).reset_index(drop=True)

    if save:
        utils.save(results, RESULTS_PATH, save_title)

def load_and_merge(athletic_df):
    """
    Loads and merges the sleep and athletic data.

    Parameters:
    athletic_df (pandas.DataFrame): The athletic dataframe.

    Returns:
    pandas.DataFrame: The merged dataframe.
    """
    sleep_df = utils.clean_load(CLEAN_PATH, 'sleep_all_period_raw.csv')
    for i in range(1, MAX_LAG):
        for column in SLEEP_FEATURE_COLUMNS:
            sleep_df[f'{column} {i}'] = sleep_df.groupby('ID')[column].shift(i)

    return sleep_df.merge(athletic_df, on=['ID', 'Date'], how='outer')

def report_spread(df, cols):
    """
    Reports the average spread for each column for each unique ID.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data.
    cols (list of str): List of columns to calculate the spread.

    Returns:
    dict: Dictionary with the average spread for each column.
    """
    return {col: df.groupby('ID')[col].std().mean() for col in cols}

if __name__ == "__main__":
    np.random.seed(0)

    hops_csv_name = f'{RAW_PATH}hops_period.csv'
    hops_df = pd.read_csv(hops_csv_name)
    hops_feature_columns = ['Vertical (no step)', 'Vertical (w/ steps)', 'CMJ']
    hops_df = clean_athletic_df(hops_df)

    hops_merged_df = load_and_merge(hops_df)
    normalization_columns = SLEEP_FEATURE_COLUMNS + hops_feature_columns
    hops_merged_df = normalize_helper(hops_merged_df, normalization_columns)

    cmj_csv_name = f'{RAW_PATH}CMJ_period.csv'
    cmj_df = pd.read_csv(cmj_csv_name, low_memory=False)
    cmj_feature_columns = ['Peak Propulsive Force (N)', 'Avg. Braking Force (N)', 'Braking Impulse (N.s)', 'Jump Height (m)', 'L|R Avg. Braking Force (%)', 'Peak Landing Force (N)', 'Peak Velocity (m/s)', 'Stiffness (N/m)']
    cmj_df = clean_athletic_df(cmj_df)

    cmj_merged_df = load_and_merge(cmj_df)
    normalization_columns = SLEEP_FEATURE_COLUMNS + cmj_feature_columns
    cmj_merged_df = normalize_helper(cmj_merged_df, normalization_columns)

    spread = report_spread(cmj_merged_df, cmj_feature_columns)
    print(spread)

    for col_of_interest in cmj_feature_columns:
        cmj_merged_min_df = cmj_merged_df.dropna(subset=[col_of_interest])
        cmj_merged_min_df = cmj_merged_min_df.loc[cmj_merged_min_df.groupby(['ID', 'Date'])[col_of_interest].idxmin()]

        cmj_save_title = f'cmj_period_sleep_regression_(min_{col_of_interest}).csv'.replace('|', '_').replace('/', ' per ')
        print(col_of_interest)
        regression_wrapper(cmj_merged_min_df, SLEEP_FEATURE_COLUMNS, cmj_feature_columns, save_title=cmj_save_title)

    shooting_csv_name = f'{RAW_PATH}shooting_period.csv'
    shooting_df = pd.read_csv(shooting_csv_name)
    shooting_feature_columns = ['Corner L%', 'Corner R%', 'Wing L%', 'Wing R%', 'Mid%', 'Overall%', 'Corner 1%', 'Corner 2%', 'Corner 3%', 'Corner 4%', 'Star 15%', 'SMS%']
    shooting_df = clean_athletic_df(shooting_df)

    shooting_merged_df = load_and_merge(shooting_df)
    normalization_columns = SLEEP_FEATURE_COLUMNS + shooting_feature_columns
    shooting_merged_df = normalize_helper(shooting_merged_df, normalization_columns)

    regression_wrapper(shooting_merged_df, SLEEP_FEATURE_COLUMNS, shooting_feature_columns, save_title='shooting_period_sleep_regression.csv')
