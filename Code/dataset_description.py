from sleep_imputation import *
from df_cleaning import *
from athletic_performance import *
from utils import *
from df_cleaning import DATA_PATH, RAW_PATH, CLEAN_PATH, OTHER_PATH, RESULTS_PATH, PLOTS_PATH, FINAL_PATH
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def dataset_description_helper(df, identifying_name):
    """
    Generates and saves a combined plot of athlete counts by academy and 
    histogram of entries by ID for the given dataframe.
    
    Parameters:
    df (pandas.DataFrame): The input dataframe.
    identifying_name (str): Identifier for the dataset used in plot titles.
    """
    df = df[df['Role'] != "Staff"]
    unique_counts = df.groupby('Academy')['ID'].nunique()

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    # First subplot: Count of Athletes by Academy
    unique_counts.plot(kind='bar', color='#750014', edgecolor='black', ax=axs[0])
    axs[0].set_title(f"Count of Athletes by Academy ({identifying_name})", fontsize=18)
    axs[0].set_ylabel('Count', fontsize=18)
    axs[0].set_xlabel('Academy', fontsize=18)
    axs[0].tick_params(axis='both', which='major', labelsize=18)

    # Second subplot: Histogram of Entries by ID
    id_counts = df['ID'].value_counts()
    axs[1].hist(id_counts, bins=20, color='#750014', edgecolor='black')
    axs[1].set_title(f'Histogram of Entries by ID ({identifying_name})', fontsize=18)
    axs[1].set_xlabel('Number of Entries', fontsize=18)
    axs[1].set_ylabel('Frequency', fontsize=18)
    axs[1].tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()

    # Save the plot
    save_path = f"{FINAL_PATH}{identifying_name}_combined_plot.png"
    fig.savefig(save_path)
    print(f'Saved to {save_path}')

if __name__ == "__main__":
    np.random.seed(0)
    
    df = utils.clean_load(CLEAN_PATH, 'sleep_all_period_raw.csv')

    feature_columns = ['HR Lowest', "HR Average", "rMSSD", "Breath Average", "Duration Integer hr"]

    print(f'Number of rows: {df.shape[0]}')
    print(f"Number of row entries with at least one NA value: {df[feature_columns].isna().any(axis=1).sum()}")
    print(f"Number of row entries with all NA values: {df[feature_columns].isna().all(axis=1).sum()}")
    print(f"Number of row entries with no NA values: {df[feature_columns].notna().all(axis=1).sum()}")
    
    role_df = df[['ID', 'Role']].drop_duplicates()
    
    shooting_csv_name = f"{RAW_PATH}shooting_period.csv"
    shooting_df = pd.read_csv(shooting_csv_name)
    shooting_df = shooting_df.merge(role_df, on='ID', how='left')

    CMJ_csv_name = f"{RAW_PATH}CMJ_period.csv"
    CMJ_df = pd.read_csv(CMJ_csv_name)
    CMJ_df = CMJ_df.merge(role_df, on='ID', how='left')
    dataset_description_helper(CMJ_df, 'Jump Dataset')

    # Display the first few rows of the dataframe
    display(df.head())

    ###### HISTOGRAM OF PROPORTION OF NIGHTS LOGGED ######
    df = df.loc[df.groupby('ID')['Days_Since_First_Logged'].idxmax()].reset_index(drop=True)
    df['Proportion_Nights_Logged'] = df['Total Nights'] / (df['Days_Since_First_Logged'] + 1)
    df = df.sort_values(by='Total Nights', ascending=True)
    display(df.head())

    total_nights_logged = df['Total Nights'].sum()
    total_days_logged = df['Days_Since_First_Logged'].sum() + len(df)
    proportion_nights_logged = total_nights_logged / total_days_logged
    print(f'The proportion of nights logged is: {proportion_nights_logged}')

    fig, ax = plt.subplots()
    ax.hist(df['Proportion_Nights_Logged'], bins=20, color='#750014', edgecolor='black')
    ax.set_title('Histogram of Proportion of Nights Logged', fontsize=16)
    ax.set_xlabel('Proportion of Nights Logged', fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.axvline(df['Proportion_Nights_Logged'].mean(), color='black', linestyle='solid', linewidth=2)
    min_ylim, max_ylim = plt.ylim()
    ax.text(df['Proportion_Nights_Logged'].mean()*1.1, max_ylim*0.9, f'Mean: {df["Proportion_Nights_Logged"].mean():.2f}', fontsize=15)
    plt.tight_layout()

    save_path = f"{FINAL_PATH}histogram_proportion_nights_logged.png"
    fig.savefig(save_path)
    print(f'Saved to {save_path}')
    ###### HISTOGRAM OF PROPORTION OF NIGHTS LOGGED ######

    #### Number of Athletes by Academy Type ####
    unique_counts = df.groupby('Academy')['ID'].nunique()
    staff_members = df[df['Role'] == "Staff"]['ID'].nunique()
    print(f'Staff Members: {staff_members}')

    bars = unique_counts.plot(kind='bar', color='#750014', edgecolor='black')
    plt.title("Number of Athletes by Academy Type")
    plt.ylabel('Count')
    plt.xlabel('Academy')
    plt.xticks(rotation=0)
    plt.tight_layout()

    for bar in bars.containers[0]:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

    save_path = f"{FINAL_PATH}barplot_academy_count.png"
    bars.figure.savefig(save_path)
    print(f'Saved to {save_path}')
    #### Number of Athletes by Academy Type ####
