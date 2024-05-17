import pandas as pd
import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import athletic_performance

from df_cleaning import DATA_PATH, RAW_PATH, CLEAN_PATH, OTHER_PATH, RESULTS_PATH, PLOTS_PATH, FINAL_PATH

class SimilarSleepers:
    """
    Class to find similar sleep values for sleep imputation
    """
    def __init__(self, df, normalization=True):
        """
        Initialize the SimilarSleepers class.

        Parameters:
        df (pandas.DataFrame): The input dataframe.
        normalization (bool): Whether to normalize the feature columns. Default is True.
        """
        self.day_of_week_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        self.feature_columns = ['HR Lowest', "HR Average", "rMSSD", "Breath Average", "Duration Integer hr"]
        self.normalization = normalization
        self.raw_df = df.copy()
        self.df = self.transform_df(df.copy())
        
    def transform_df(self, df):
        """
        Transforms the dataframe for matrix sleep imputation.
        Normalizes the feature columns if self.normalization is True.

        Parameters:
        df (pandas.DataFrame): The input dataframe.

        Returns:
        pandas.DataFrame: The transformed dataframe.
        """
        if self.normalization:
            df = athletic_performance.normalize_helper(df, self.feature_columns)
       
        df['Day_of_Week'] = df['Date'].dt.day_name()
        df['Day_Of_Week_First_Logged'] = df['Date_First_Logged'].dt.day_name()
        df['Day_Of_Week_First_Logged'] = df['Day_Of_Week_First_Logged'].map(self.day_of_week_map)
        df['Week_Since_First_Logged'] = (df['Days_Since_First_Logged'] + df['Day_Of_Week_First_Logged']) // 7 
        df['Adjusted_Days_Since_First_Logged'] = df['Week_Since_First_Logged'] * 7 + df['Day_of_Week'].map(self.day_of_week_map)
        
        utils.save(df, RESULTS_PATH, "similar_sleepers.csv")
        return df

    def heatmap(self, feature):
        """
        Plots a heatmap of the specified feature showing the presence of data entries.

        Parameters:
        feature (str): The feature to plot in the heatmap.
        """
        df = self.df.copy()

        top_20_ids = df.groupby('ID')['Total Nights'].sum().sort_values(ascending=False).head(20).index
        df = df[df['ID'].isin(top_20_ids)]
        
        pivot = df.pivot_table(index="ID", columns="Adjusted_Days_Since_First_Logged", values=feature, aggfunc="count")
        binary_pivot = pivot.notnull().astype('int')
        binary_pivot = binary_pivot.sort_values(by='ID', ascending=False)

        plt.figure(figsize=(15, 10))
        cmap = sns.color_palette(["white", "#750014"])
        sns.heatmap(binary_pivot, cbar=False, cmap=cmap, xticklabels=1)
        
        max_days = df['Adjusted_Days_Since_First_Logged'].max()
        tick_frequency = 100
        
        plt.xlabel("Days Since First Night Logged", fontsize=20)
        plt.ylabel("ID", fontsize=20)
        plt.title(f"Logged Nights of Sleep By Player (Red Tick for Logged Night)", fontsize=20)
        plt.xticks(ticks=np.arange(0, max_days+1, tick_frequency), labels=np.arange(0, max_days+1, tick_frequency), fontsize=18, rotation=0)
        plt.yticks(fontsize=18, rotation=0)
        
        save_path = f"{FINAL_PATH}sleep_heatmap.png"
        plt.savefig(save_path)
        plt.show()
        print(f'Saved to {save_path}')

    def matrix_by_weekday(self):
        """
        Creates a matrix where each ID has 21 columns per feature column. The columns represent the mean, 
        standard deviation, and count of nights logged for each day of the week.

        Returns:
        pandas.DataFrame: The matrix dataframe.
        """
        df = self.df.copy()
        matrix = pd.DataFrame()

        for id in df['ID'].unique():
            individual_row_entry = {"ID": id}
            for feature in self.feature_columns:
                filtered_df = df[df['ID'] == id][feature].dropna()
                individual_row_entry[f"{feature}_mean"] = filtered_df.mean()
                individual_row_entry[f"{feature}_std"] = filtered_df.std()
                individual_row_entry[f"{feature}_count"] = filtered_df.count()

                for day in self.day_of_week_map.keys():
                    day_filtered_df = df[(df['ID'] == id) & (df['Day_of_Week'] == day)][feature].dropna()
                    individual_row_entry[f"{feature}_{day}_mean"] = day_filtered_df.mean()
                    individual_row_entry[f"{feature}_{day}_std"] = day_filtered_df.std()
                    individual_row_entry[f"{feature}_{day}_count"] = day_filtered_df.count()
            
            matrix = pd.concat([matrix, pd.DataFrame(individual_row_entry, index=[0])], ignore_index=True)

        return matrix

if __name__ == "__main__":
    np.random.seed(0)
    df = utils.clean_load(CLEAN_PATH, 'sleep_all_period_raw.csv')
    display(df.head())

    similar_sleepers = SimilarSleepers(df)
    
    # Heatmap
    test_feature = "HR Average"
    similar_sleepers.heatmap(test_feature)

    # Matrix by weekday
    weekday_matrix = similar_sleepers.matrix_by_weekday()
    display(weekday_matrix.head())
    utils.save(weekday_matrix, RESULTS_PATH, "matrix_by_weekday.csv")
