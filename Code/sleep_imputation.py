import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from df_cleaning import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from df_cleaning import DATA_PATH, RAW_PATH, CLEAN_PATH, OTHER_PATH, RESULTS_PATH, PLOTS_PATH, FINAL_PATH


np.random.seed(0)

class IndividualAnalysis:
    """
    Class to perform analysis on a dataframe
    """
    def __init__(self, df, id, normalization=True):
        """
        Initialize the IndividualAnalysis class.

        Parameters:
        df (pandas.DataFrame): The input dataframe.
        id (int): The ID of the individual.
        normalization (bool): Whether to normalize the feature columns. Default is True.
        """
        self.index = "Days_Since_First_Logged"
        self.feature_columns = ['HR Lowest', "HR Average", "rMSSD", "Breath Average", "Duration Integer hr"]
        self.id = id
        self.normalization = normalization
        self.raw_df = df.copy()

        self.df = df.copy().dropna()
        self.df = self.df[self.df['ID'] == id]
        self.df = self.df[[self.index] + self.feature_columns]
        self.df.set_index(self.index, inplace=True)

        # Create an entry for all days in the range (0, max_days + 1). If sleep wasn't logged, mark it as a blank line.
        max_days = self.df.index.max()
        all_days = pd.Index(np.arange(0, max_days + 1), name=self.index)
        self.df = self.df.reindex(all_days)

        self.most_consecutive_df = self.most_consecutive_entries()
        self.train_df, self.test_df = self.randomized_train_test_split(self.df)

        # Sort df, train_df, and test_df by index
        self.df = self.df.sort_index()
        self.train_df = self.train_df.sort_index()
        self.test_df = self.test_df.sort_index()

        self.results_df_columns = ["method", "num_points_fit", "min_points_fit", "window", "backwards", "weights", "id", "nights logged", "test points", "feature", "rmse", "average_weighting"]
        self.empty_results_df = pd.DataFrame(columns=self.results_df_columns)

    def randomized_train_test_split(self, df, train_size=0.7):
        """
        Randomly drop out data from df to create a train and test set (NaNs are dropped).

        Parameters:
        df (pandas.DataFrame): The input dataframe.
        train_size (float): The percentage of data to use for training.

        Returns:
        tuple: The train and test dataframes.
        """
        df = df.copy().dropna()
        train = df.sample(frac=train_size, random_state=0)
        test = df.drop(train.index)
        return train, test

    def most_consecutive_entries(self):
        """
        Returns the subset of df with the most consecutive entries (no NaNs).
        """
        df = self.df.copy()
        mask = df.notna().all(axis=1)
        df['group'] = (~mask).cumsum()
        counts = df[mask].groupby('group').size()
        max_group = counts.idxmax()
        df = df[df['group'] == max_group]
        df = df.drop(columns='group')
        return df

    def presence_chart(self):
        """
        Plots a line chart where "Days_Since_First_Logged" is the x-axis and y is 0 if there is no data for that day and 1 if there is data for that day.
        """
        df_reset = self.df.copy().reset_index()
        y = df_reset[self.feature_columns[0]].notnull().astype('int')

        plt.figure(figsize=(10, 5))
        plt.plot(df_reset[self.index], y)
        plt.title(f"Presence of entry (ID = {self.id})")
        plt.ylabel('Presence of entry')
        plt.xlabel('Days Since First Logged')
        plt.show()

    def plot_feature(self, X, Y, feature):
        """
        Plots a line chart of feature (y-axis) vs the index column (x-axis, days since first logged).
        If there is no data for a day (NaN), then display a gap in the line chart.

        Parameters:
        X (numpy array): x values for all data.
        Y (numpy array): y values for all data.
        feature (str): The feature to plot.
        """
        Y = Y[:, self.feature_columns.index(feature)].reshape(-1, 1)

        plt.figure(figsize=(10, 5))
        plt.plot(X, Y)
        plt.title(f"{feature} vs Days Since First Logged (ID = {self.id})")
        plt.ylabel(feature)
        plt.xlabel('Days Since First Logged')
        plt.show()
    
    def plot_all_features(self):
        """
        Plot all features as a line chart in one figure.
        """
        dataset = self.df.copy()
        values = dataset.values
        groups = [i for i in range(len(self.feature_columns))]
        plt.figure(figsize=(10, 8))
        for i, group in enumerate(groups):
            plt.subplot(len(groups), 1, i + 1)
            plt.plot(values[:, group])
            plt.title(dataset.columns[group], y=0.5, loc='right')
        plt.show()

    def histogram_residuals(self, residuals, title):
        """
        Plot a histogram of the residuals.

        Parameters:
        residuals (numpy array): The residuals.
        title (str): The title of the plot.
        """
        plt.hist(residuals, bins=20, edgecolor='black')
        plt.title(title)
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.show()

    def plot_helper(self, title, X, Y, test_X, test_Y, y_pred, x_axis_label, y_axis_label):
        """
        Plot the actual vs predicted values.

        Parameters:
        title (str): The title of the plot.
        X (numpy array): x values for all data.
        Y (numpy array): y values for all data.
        test_X (numpy array): x values for test data.
        test_Y (numpy array): y values for test data.
        y_pred (numpy array): Predicted values for test data.
        x_axis_label (str): Label for x-axis.
        y_axis_label (str): Label for y-axis.
        """
        plt.plot(X, Y, color="black", label="all data", linestyle='--', linewidth=0.2)
        plt.scatter(test_X, test_Y, color="red", label="test data", s=10)
        plt.scatter(test_X, y_pred, color="navy", label="prediction", s=10)
        for i in range(len(test_X)):
            plt.plot([test_X[i], test_X[i]], [test_Y[i], y_pred[i]], color="green", linewidth=1)
        plt.axis("tight")
        plt.legend()
        plt.title(title)
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.tight_layout()
        save_path = RESULTS_PATH + f"/knn_plots/{y_axis_label}.png"
        plt.savefig(save_path)
        plt.clf()

    def normalization_helper(self, to_fit, to_transform, type):
        """
        Fit a scaler to to_fit and transform to_transform.

        Parameters:
        to_fit (numpy array): Data to fit the scaler.
        to_transform (numpy array): Data to transform with the scaler.
        type (str): Type of scaler to use (i.e. "minmax").

        Returns:
        numpy array: The transformed data.
        """
        if type == "minmax":
            scaler = MinMaxScaler()
            scaler.fit(to_fit)
            transformed = scaler.transform(to_transform)
        elif type == "mean":
            mean = np.mean(to_fit, axis=0)
            transformed = to_transform / mean
        else:
            raise Exception(f"Invalid scaler type: {type}")
        
        return transformed
    
    def check_valid_lin_point(self, train_X, point_X, lin_window, train_Y=None):
        """
        Check if point_X is a valid point for linear interpolation.

        Parameters:
        train_X (numpy array): x values for training data.
        point_X (numpy array): x value for test data.
        lin_window (int): Window size for linear interpolation.
        train_Y (numpy array): y values for training data.

        Returns:
        bool: True if point_X is a valid point for linear interpolation.
        """
        valid_lin_point = True
        less_than_points = train_X[train_X < point_X]
        greater_than_points = train_X[train_X > point_X]
        if len(less_than_points) == 0 or len(greater_than_points) == 0:
            valid_lin_point = False
        else:
            closest_less_than_x = less_than_points[-1]
            closest_greater_than_x = greater_than_points[0]
            if train_Y is not None:
                closest_less_than_y = train_Y[np.where(train_X == closest_less_than_x)[0][0]]
                closest_greater_than_y = train_Y[np.where(train_X == closest_greater_than_x)[0][0]]
            if lin_window != float('inf'):
                if point_X - closest_less_than_x > lin_window or closest_greater_than_x - point_X > lin_window:
                    valid_lin_point = False
        
        return valid_lin_point

    def check_valid_knn_point(self, train_X, point_X, knn_window, knn_backwards, knn_n_neighbors, train_Y=None):
        """
        Check if point_X is a valid point for KNN.

        Parameters:
        train_X (numpy array): x values for training data.
        point_X (numpy array): x value for test data.
        knn_window (int): Window size for KNN.
        knn_backwards (bool): Whether to use backward KNN.
        knn_n_neighbors (int): Number of neighbors to use.
        train_Y (numpy array): y values for training data.

        Returns:
        bool: True if point_X is a valid point for KNN.
        """
        valid_knn_point = True
        if knn_window == float('inf'):
            mask = train_X < point_X if knn_backwards else np.ones(len(train_X), dtype=bool)
        else:
            upper = point_X if knn_backwards else point_X + knn_window
            lower = point_X - knn_window
            nights = np.arange(lower[0], upper[0])
            mask = np.isin(train_X, nights)

        training_points = len(train_X[mask])
        if training_points < knn_n_neighbors:
            valid_knn_point = False

        return valid_knn_point

    def check_valid_quadratic_linear_point(self, train_X, point_X, quadratic_linear_window, min_train_points, train_Y=None):
        """
        Check if point_X is a valid point for quadratic/linear fit.

        Parameters:
        train_X (numpy array): x values for training data.
        point_X (numpy array): x value for test data.
        quadratic_linear_window (int): Window size for quadratic/linear fit.
        min_train_points (int): Minimum number of points to use for fitting.
        train_Y (numpy array): y values for training data.

        Returns:
        bool: True if point_X is a valid point for quadratic/linear fit.
        """
        closest_less_than_x = train_X[(train_X < point_X) & (train_X >= point_X - quadratic_linear_window)]
        valid_quadratic_linear_point = len(closest_less_than_x) >= min_train_points

        return valid_quadratic_linear_point

    def determine_valid_test_indices(self, lin_parameters=None, knn_parameters=None, quadratic_linear_parameters=None, plot=False):
        """
        Determine the valid test indices based on the parameters.

        Parameters:
        lin_parameters (dict): Parameters for linear interpolation.
        knn_parameters (dict): Parameters for KNN.
        quadratic_linear_parameters (dict): Parameters for quadratic or linear fit.
        plot (bool): Whether to plot the results.

        Returns:
        tuple: The X, Y numpy arrays and the filtered test indices.
        """
        df = self.df.copy().dropna()
        X = df.index.to_numpy().reshape(-1, 1)
        Y = df.to_numpy()

        def filter_data(X, Y, lin_parameters, knn_parameters, quadratic_linear_parameters):
            filtered_test_indices = []
            for i in range(len(X)):
                point_X = X[i]
                train_X, train_Y = self.remove_point(i, X, Y)
                valid_lin_point = valid_knn_point = valid_quadratic_linear_point = True

                if lin_parameters:
                    lin_window = lin_parameters["window"]
                    valid_lin_point = self.check_valid_lin_point(train_X, point_X, lin_window)

                if knn_parameters:
                    knn_window = knn_parameters["window"]
                    knn_backwards = knn_parameters["backwards"]
                    knn_n_neighbors = knn_parameters["n_neighbors"]
                    valid_knn_point = self.check_valid_knn_point(train_X, point_X, knn_window, knn_backwards, knn_n_neighbors)

                if quadratic_linear_parameters:
                    quadratic_linear_window = quadratic_linear_parameters["window"]
                    quadratic_linear_min_train_points = quadratic_linear_parameters["min_train_points"]
                    valid_quadratic_linear_point = self.check_valid_quadratic_linear_point(train_X, point_X, quadratic_linear_window, quadratic_linear_min_train_points)

                if valid_lin_point and valid_knn_point and valid_quadratic_linear_point:
                    filtered_test_indices.append(i)

            return np.array(filtered_test_indices)

        filtered_test_indices = filter_data(X, Y, lin_parameters, knn_parameters, quadratic_linear_parameters)
        if self.normalization:
            Y = self.normalization_helper(Y, Y, "mean")

        return X, Y, filtered_test_indices

    def KNN(self, X, Y, test_indices, feature, backwards, window=float('inf'), n_neighbors=5, plot=False, average_weighting=0):
        """
        Use KNN to predict a feature.

        Parameters:
        X (numpy array): x values for all data.
        Y (numpy array): y values for all data.
        test_indices (numpy array): Indices of data points to test.
        feature (str): The feature to predict (i.e. 'HR Average').
        backwards (bool): Whether to use backward KNN.
        window (int): Window of nights to use.
        n_neighbors (int): Number of neighbors to use.
        plot (bool): Whether to plot the results.
        average_weighting (float): Weight of the average value.

        Returns:
        pandas.DataFrame: The results dataframe.
        """
        test_X = X[test_indices]
        test_Y = Y[test_indices]
        test_Y = test_Y[:, self.feature_columns.index(feature)].reshape(-1, 1)
        Y = Y[:, self.feature_columns.index(feature)].reshape(-1, 1)

        results_df = self.empty_results_df.copy()
        weights_to_examine = ["uniform", "distance"]

        for weight_index, weights in enumerate(weights_to_examine):
            predictions = []
            modified_test_X = []
            modified_test_Y = []
            for i in test_indices:
                train_X, train_Y = self.remove_point(i, X, Y, backwards)
                point = X[i]
                valid_knn_point, filtered_train_X, filtered_train_Y = self.check_valid_knn_point(train_X, point, window, backwards, n_neighbors, train_Y)
                if valid_knn_point:
                    knn = KNeighborsRegressor(n_neighbors, weights=weights)
                    knn.fit(filtered_train_X, filtered_train_Y)
                    prediction = knn.predict(point.reshape(1, -1))[0]
                    if average_weighting > 0:
                        prediction = self.blend_daily_average(average_weighting, prediction, train_Y, backwards)
                    predictions.append(prediction)
                    modified_test_Y.append(test_Y[np.where(test_X == point)[0][0]])
                    modified_test_X.append(point)
                else:
                    print(f"Point {point} is not a valid KNN point. \n parameters: window = {window}, backwards = {backwards}, n_neighbors = {n_neighbors}")

            predictions = np.array(predictions)
            if len(predictions) == 0:
                return results_df

            method = "KNN"
            num_points_fit = n_neighbors
            min_points_fit = n_neighbors
            id_ = self.id
            nights_logged = self.raw_df[self.raw_df['ID'] == self.id].shape[0]
            test_points = len(modified_test_X)
            residuals = modified_test_Y - predictions  
            rmse = round(sqrt(mean_squared_error(modified_test_Y, predictions)), 5)
            results_df.loc[weight_index] = [method, num_points_fit, min_points_fit, window, backwards, weights, id_, nights_logged, test_points, feature, rmse, average_weighting]

            if plot:
                title = f"KNN (backward = {backwards}, k = {n_neighbors}, window = {window}, weights = '{weights}') \n feature = {feature}, id = {self.id}, nights logged = {nights_logged}, \n RMSE = {rmse}"
                x_axis_label = "Days Since First Logged"
                y_axis_label = feature
                if self.normalization:
                    y_axis_label += " (normalized)"
                self.plot_helper(title, X, Y, modified_test_X, modified_test_Y, predictions, x_axis_label, y_axis_label)
                self.histogram_residuals(residuals, title)

        return results_df
    
    def linear_interpolation(self, X, Y, test_indices, feature, window, plot=False, average_weighting=0):
        """
        Use linear interpolation to predict a feature.

        Parameters:
        X (numpy array): x values for all data.
        Y (numpy array): y values for all data.
        test_indices (numpy array): Indices of data points to test.
        feature (str): The feature to predict (i.e. 'HR Average').
        window (int): Window of nights to use.
        plot (bool): Whether to plot the results.
        average_weighting (float): Weight of the average value.

        Returns:
        pandas.DataFrame: The results dataframe.
        """
        Y = Y[:, self.feature_columns.index(feature)].reshape(-1, 1)
        test_X = X[test_indices]
        test_Y = Y[test_indices]

        predictions_test = []
        modified_test_Y = []
        modified_test_X = []
        for i in test_indices:
            train_X, train_Y = self.remove_point(i, X, Y)
            point = X[i]
            valid_lin_point, closest_less_than_x, closest_greater_than_x, closest_less_than_y, closest_greater_than_y = self.check_valid_lin_point(train_X, point, window, train_Y)
            if valid_lin_point:
                slope = (closest_greater_than_y - closest_less_than_y) / (closest_greater_than_x - closest_less_than_x)
                y_pred = slope * (point - closest_less_than_x) + closest_less_than_y
                y_pred = np.array(y_pred).reshape(-1, 1)
                if average_weighting > 0:
                    y_pred = self.blend_daily_average(average_weighting, y_pred, train_Y, backwards=False)
                predictions_test.append(y_pred)
                modified_test_Y.append(test_Y[np.where(test_X == point)[0][0]])
                modified_test_X.append(point)
            else:
                print(f"Point {point} is not a valid linear interpolation point!!! Parameters: window = {window}")
        
        predictions_test = np.array(predictions_test)
        modified_test_Y = np.array(modified_test_Y).reshape(-1)
        modified_test_X = np.array(modified_test_X).reshape(-1)

        results_df = self.empty_results_df.copy()
        method = "lin"
        num_points_fit = 2
        min_points_fit = 2
        id_ = self.id
        nights_logged = self.raw_df[self.raw_df['ID'] == self.id].shape[0]
        test_points = len(modified_test_X)
        residuals = modified_test_Y - predictions_test
        rmse = round(sqrt(mean_squared_error(modified_test_Y, predictions_test)), 5)
        results_df.loc[0] = [method, num_points_fit, min_points_fit, window, False, None, id_, nights_logged, test_points, feature, rmse, average_weighting]

        if plot:
            title = f"Linear Interpolation (window = {window}) \n feature = {feature}, id = {self.id}, nights logged = {nights_logged}, \n RMSE = {rmse}"
            x_axis_label = "Days Since First Logged"
            y_axis_label = feature
            if self.normalization:
                y_axis_label += " (normalized)"
            self.plot_helper(title, X, Y, modified_test_X, modified_test_Y, predictions_test, x_axis_label, y_axis_label)
            self.histogram_residuals(residuals, title)

        return results_df
    
    def quadratic_linear_fit(self, X, Y, test_indices, feature, window, min_train_points=3, plot=False, degree=2, sanity_check=False, average_weighting=0):
        """
        Use quadratic or linear fit to predict a feature.

        Parameters:
        X (numpy array): x values for all data.
        Y (numpy array): y values for all data.
        test_indices (numpy array): Indices of data points to test.
        feature (str): The feature to predict (i.e. 'HR Average').
        window (int): Window of nights to use.
        min_train_points (int): Minimum number of points to use for fitting.
        plot (bool): Whether to plot the results.
        degree (int): Degree of the polynomial to use (1 for linear, 2 for quadratic).
        sanity_check (bool): Whether to plot the results at the individual point level for sanity check.
        average_weighting (float): Weight of the average value.

        Returns:
        pandas.DataFrame: The results dataframe.
        """
        assert degree in [1, 2], "Degree must be 1 or 2"

        test_X = X[test_indices]
        test_Y = Y[test_indices]
        test_Y = test_Y[:, self.feature_columns.index(feature)].reshape(-1, 1)
        Y = Y[:, self.feature_columns.index(feature)].reshape(-1, 1)

        predictions_test = []
        modified_test_Y = []
        modified_test_X = []
        for i in test_indices:
            train_X, train_Y = self.remove_point(i, X, Y, backwards=True)
            point = X[i]
            valid_point, closest_less_than_x, closest_less_than_y = self.check_valid_quadratic_linear_point(train_X, point, window, min_train_points, train_Y)
            if valid_point:
                if degree == 2:
                    quadratic = np.polyfit(closest_less_than_x, closest_less_than_y, 2)
                    y_pred = quadratic[0] * point**2 + quadratic[1] * point + quadratic[2]
                else:
                    linear = np.polyfit(closest_less_than_x, closest_less_than_y, 1)
                    y_pred = linear[0] * point + linear[1]

                if sanity_check:
                    plt.plot(closest_less_than_x, closest_less_than_y, color="black", label="closest_less_than_y", linestyle='--', linewidth=0.4)
                    plt.plot(point, y_pred, 'o', color="navy", label="y_pred")
                    plt.plot(point, test_Y[np.where(test_X == point)[0][0]], 'o', color="red", label="actual point")
                    train_and_test = np.concatenate((closest_less_than_x, point), axis=0)
                    if degree == 1:
                        plt.plot(train_and_test, linear[0] * train_and_test + linear[1], color="blue", label="linear fit")
                    else:
                        plt.plot(train_and_test, quadratic[0] * train_and_test**2 + quadratic[1] * train_and_test + quadratic[2], color="blue", label="quadratic fit")
                    plt.axis("tight")
                    plt.legend()
                    plt.title(f"Sanity Check (ID = {self.id})")
                    plt.xlabel('Days Since First Logged')
                    plt.ylabel(feature)
                    plt.tight_layout()
                    plt.show()

                if average_weighting > 0:
                    y_pred = self.blend_daily_average(average_weighting, y_pred, train_Y, backwards=True)
                y_pred = np.array(y_pred).reshape(-1, 1)
                predictions_test.append(y_pred)
                modified_test_Y.append(test_Y[np.where(test_X == point)[0][0]])
                modified_test_X.append(point)
            else:
                print(f"Point {point} is not a valid quadratic point!!!. Parameters: window = {window}, min_train_points = {min_train_points}")
                
        predictions_test = np.array(predictions_test)
        modified_test_Y = np.array(modified_test_Y).reshape(-1)
        modified_test_X = np.array(modified_test_X).reshape(-1)

        results_df = self.empty_results_df.copy()
        method = "Quadratic" if degree == 2 else "Linear"
        num_points_fit = window if min_train_points == window else None
        id_ = self.id
        nights_logged = self.raw_df[self.raw_df['ID'] == self.id].shape[0]
        test_points = len(modified_test_X)
        residuals = modified_test_Y - predictions_test
        rmse = round(sqrt(mean_squared_error(modified_test_Y, predictions_test)), 5)
        results_df.loc[0] = [method, num_points_fit, min_train_points, window, True, None, id_, nights_logged, test_points, feature, rmse, average_weighting]
    
        if plot:
            title = f"{method} (window = {window}) \n feature = {feature}, id = {self.id}, nights logged = {nights_logged}, \n RMSE = {rmse}"
            x_axis_label = "Days Since First Logged"
            y_axis_label = feature
            if self.normalization:
                y_axis_label += " (normalized)"
            self.plot_helper(title, X, Y, modified_test_X, modified_test_Y, predictions_test, x_axis_label, y_axis_label)
            self.histogram_residuals(residuals, title)

        return results_df
    
    def remove_point(self, index, x_array, y_array, backwards=False):
        """
        Remove a point from the x_array and y_array.

        Parameters:
        index (int): Index of the point to remove.
        x_array (numpy array): x values.
        y_array (numpy array): y values.

        Returns:
        tuple: The modified x_array and y_array.
        """
        x_array = x_array.copy()
        y_array = y_array.copy()
        if not backwards:
            x_array = np.delete(x_array, index, axis=0)
            y_array = np.delete(y_array, index, axis=0)
        else:
            x_array = x_array[:index]
            y_array = y_array[:index]
        return x_array, y_array
    
    def determine_average(self, train_Y, backwards=False):
        """
        Determine the average of the feature.

        Parameters:
        train_Y (numpy array): y values for training data.
        backwards (bool): Whether to only look backwards.

        Returns:
        float: The average value.
        """
        return np.mean(train_Y)

    def blend_daily_average(self, average_weighting, predicted_value, train_Y, backwards):
        """
        Blend the predicted value with the daily average.

        Parameters:
        average_weighting (float): Weight of the average value.
        predicted_value (numpy array): Predicted value.
        train_Y (numpy array): y values for training data.
        backwards (bool): Whether to only look backwards.

        Returns:
        numpy array: The blended value.
        """
        average = self.determine_average(train_Y, backwards)
        blended_value = average_weighting * average + (1 - average_weighting) * predicted_value[0]
        return np.array([blended_value])

def analyze_individual(df, id_, normalization, feature, lin_parameters=None, knn_parameters=None, quadratic_linear_parameters=None, plot=True):
    """
    Perform analysis on an individual.

    Parameters:
    df (pandas.DataFrame): The dataset.
    id_ (int): The ID of the individual.
    normalization (bool): Whether to normalize the data.
    feature (str): The feature to plot.
    lin_parameters (dict): Parameters for linear interpolation.
    knn_parameters (dict): Parameters for KNN.
    quadratic_linear_parameters (dict): Parameters for quadratic and linear fit.
    plot (bool): Whether to plot the results.
    """
    individual = IndividualAnalysis(df, id_, normalization)
    X, Y, test_indices = individual.determine_valid_test_indices(lin_parameters, knn_parameters, quadratic_linear_parameters)
    if plot:
        individual.plot_feature(X, Y, feature)
    total_points = X.shape[0]
    test_points = test_indices.shape[0]
    percent_valid = test_points / total_points * 100
    print(f'valid test points: {test_points} / {total_points} ({percent_valid:.1f}%)')

    results = pd.DataFrame()
    for average_weighting in [0, 0.5, 1]:
        if lin_parameters:
            lin_window = lin_parameters['window']
            individual_lin_results = individual.linear_interpolation(X, Y, test_indices, feature, lin_window, plot=plot, average_weighting=average_weighting)

        if knn_parameters:
            knn_window = knn_parameters['window']
            knn_n_neighbors = knn_parameters['n_neighbors']
            knn_backwards = knn_parameters['backwards']
            individual_knn_results = individual.KNN(X, Y, test_indices, feature, knn_backwards, knn_window, knn_n_neighbors, plot=plot, average_weighting=average_weighting)
        
        if quadratic_linear_parameters:
            quad_linear_window = quadratic_linear_parameters['window']
            quad_linear_min_train_points = quadratic_linear_parameters['min_train_points']
            individual_quadratic_results = individual.quadratic_linear_fit(X, Y, test_indices, feature, quad_linear_window, quad_linear_min_train_points, plot=plot, degree=2, average_weighting=average_weighting)
            individual_linear_results = individual.quadratic_linear_fit(X, Y, test_indices, feature, quad_linear_window, quad_linear_min_train_points, plot=plot, degree=1, average_weighting=average_weighting)

        if results.empty:
            results = pd.concat([individual_lin_results, individual_knn_results, individual_quadratic_results, individual_linear_results], ignore_index=True)
        else:
            results = pd.concat([results, individual_lin_results, individual_knn_results, individual_quadratic_results, individual_linear_results], ignore_index=True)

    results = results.sort_values(by=['rmse'])
    display(results)

def full_analysis(df):
    """
    Perform full analysis on all individuals and save results to CSV.

    Parameters:
    df (pandas.DataFrame): The dataset.
    """
    ids = df['ID'].value_counts().index.tolist()
    full_knn_results = pd.DataFrame()
    full_lin_results = pd.DataFrame()
    full_quadratic_results = pd.DataFrame()
    full_linear_results = pd.DataFrame()

    min_test_points = 100
    min_test_split = 0.2

    for id in ids:
        if df[df['ID'] == id].shape[0] > min_test_points:
            analysis = IndividualAnalysis(df, id, normalization=True)
            plot = False
            min_lin_window = 1
            max_lin_window = 10
            max_knn_window = 10
            min_knn_window = 2
            window_to_neighbor = 1
            max_n_neighbors = max_knn_window // window_to_neighbor
            limiting_knn_window = max_n_neighbors
            most_restrictive_backwards = True
            max_quad_linear_window = 15
            min_quad_linear_window = 3

            X, Y, test_indices = analysis.determine_valid_test_indices(
                lin_parameters={"window": min_lin_window},
                knn_parameters={"window": limiting_knn_window, "backwards": most_restrictive_backwards, "n_neighbors": max_n_neighbors},
                quadratic_linear_parameters={"window": max_quad_linear_window, "min_train_points": max_quad_linear_window})
            
            total_points = X.shape[0]
            test_points = test_indices.shape[0]
            percent_valid = test_points / total_points * 100

            if test_points > min_test_points and percent_valid > min_test_split * 100:
                print(f'{id} valid test points: {test_points} / {total_points} ({percent_valid:.1f}%)')
                for feature in analysis.feature_columns:
                    for average_weighting in [0, 0.5, 1]:
                        step = 1
                        for lin_window in range(min_lin_window, max_lin_window + 1, step):
                            individual_lin_results = analysis.linear_interpolation(X, Y, test_indices, feature, lin_window, plot=plot, average_weighting=average_weighting)
                            if full_lin_results.empty:
                                full_lin_results = individual_lin_results
                            else:
                                full_lin_results = pd.concat([full_lin_results, individual_lin_results], ignore_index=True)
                        
                        for window in range(min_knn_window, max_knn_window + 1, window_to_neighbor):
                            n_neighbors = window // window_to_neighbor
                            if n_neighbors > 0:
                                for backwards in [True, False]:
                                    individual_knn_results = analysis.KNN(X, Y, test_indices, feature, backwards=backwards, window=window, n_neighbors=n_neighbors, plot=plot, average_weighting=average_weighting)
                                    if full_knn_results.empty:
                                        full_knn_results = individual_knn_results
                                    else:
                                        full_knn_results = pd.concat([full_knn_results, individual_knn_results], ignore_index=True)
                        
                        for window in range(min_quad_linear_window, max_quad_linear_window + 1, step):
                            individual_quadratic_results = analysis.quadratic_linear_fit(X, Y, test_indices, feature, window=window, min_train_points=window, plot=plot, degree=2, average_weighting=average_weighting)
                            if full_quadratic_results.empty:
                                full_quadratic_results = individual_quadratic_results
                            else:
                                full_quadratic_results = pd.concat([full_quadratic_results, individual_quadratic_results], ignore_index=True)
            
                            individual_linear_results = analysis.quadratic_linear_fit(X, Y, test_indices, feature, window=window, min_train_points=window, plot=plot, degree=1, average_weighting=average_weighting)
                            if full_linear_results.empty:
                                full_linear_results = individual_linear_results
                            else:
                                full_linear_results = pd.concat([full_linear_results, individual_linear_results], ignore_index=True)
                
    full_lin_results = full_lin_results.sort_values(by=['rmse'])
    utils.save(full_lin_results, RESULTS_PATH, 'full_lin_results.csv') 
    display(full_lin_results)
                    
    full_knn_results = full_knn_results.sort_values(by=['rmse'])
    utils.save(full_knn_results, RESULTS_PATH, 'full_knn_results.csv')
    display(full_knn_results)

    full_quadratic_results = full_quadratic_results.sort_values(by=['rmse'])
    utils.save(full_quadratic_results, RESULTS_PATH, 'full_quadratic_results.csv')
    display(full_quadratic_results)

    full_linear_results = full_linear_results.sort_values(by=['rmse'])
    utils.save(full_linear_results, RESULTS_PATH, 'full_linear_results.csv')
    display(full_linear_results)

    all_results = pd.concat([full_lin_results, full_knn_results, full_quadratic_results, full_linear_results], ignore_index=True)
    all_results = all_results.sort_values(by=['rmse'])
    utils.save(all_results, RESULTS_PATH, 'all_results.csv')
    display(all_results)


if __name__ == '__main__':
    np.random.seed(0)
    df = utils.clean_load(CLEAN_PATH, 'sleep_all_period_raw.csv')

    id_ = 47232
    normalization = True
    plot = True
    feature = "HR Average"
    lin_parameters = {"window": 1}
    knn_parameters = {"window": 8, "backwards": True, "n_neighbors": 8}
    quadratic_linear_parameters = {"window": 3, "min_train_points": 3}
    analyze_individual(df, id_, normalization, feature, lin_parameters, knn_parameters, quadratic_linear_parameters, plot=plot)

    full_analysis(df)
