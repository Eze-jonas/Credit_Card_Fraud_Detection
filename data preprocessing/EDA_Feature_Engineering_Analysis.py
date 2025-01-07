# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:01:36 2024

@author: USER
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
import os

# import and upload the cleaned dataset
# specify file path
file_path = r"C:\Users\USER\Desktop\credit_card_fraud_detection\cleaned_data.csv"
# upload the data into a datafram df
df = pd.read_csv(file_path)
print(df.head)


# Examine linear relationship among the variables using correlation matrix and heatmap
def plot_correlation_heatmap(df, figsize=(10, 5), cmap="coolwarm", annot=False):
    """
    Plots a correlation matrix heatmap for a given DataFrame.

    Parameters:
    -df (pd.DataFrame): The input DataFrame.
    -figsize (tuple): Figure size for the heatmap. Default is (10, 5).
    -cmap (str): Colormap for the heatmap. Default is "coolwarm".
    -annot (bool): If True, display correlation values in each cell. Default is False.
    """
    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Set up the figure size and heatmap styling
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, linewidths=0.5)

    # Add title and show the plot
    plt.title("Correlation Matrix Heatmap")
    plt.show()


# Call the function
plot_correlation_heatmap(df)


# Add time difference column
def add_time_difference(df, time_col='Time'):
    """
    Adds a 'Time Difference' column to the DataFrame based on the difference
    between consecutive values in the specified time column. Places the new
    column next to the time column if it's added.

    Parameters:
    -df (pd.DataFrame): The DataFrame containing the time column.
    -time_col (str): The name of the time column to use for difference calculation. Default is 'Time'.

    Returns:
    -pd.DataFrame: The modified DataFrame with the 'Time Difference' column added.
    """
    # Check if 'Time Difference' already exists, if not, calculate it
    if 'Time Difference' not in df.columns:
        # Calculate the time difference and add the column
        df['Time Difference'] = df[time_col].diff().fillna(0)  # Fill first NaN with 0

    # Move the 'Time Difference' column next to the specified time column
    columns = list(df.columns)
    if 'Time Difference' in columns:
        columns.remove('Time Difference')
        time_index = columns.index(time_col) + 1
        columns.insert(time_index, 'Time Difference')
    df = df[columns]

    return df


# Call the function
df = add_time_difference(df)
print(df)

# Examine the linear relationship between the time difference column and other columns
plot_correlation_heatmap(df)


# Examine the distribution of the target variable(Class column) using a histogram

def plot_target_distribution(df, target_col='Class', color='skyblue', figsize=(8, 5)):
    """
    Plots a histogram for the target variable with counts annotated on each bar.

    Parameters:
    -df (pd.DataFrame): The DataFrame containing the target column.
    -target_col (str): The column name of the target variable. Default is 'Class'.
    -color (str): Color for the bars in the histogram. Default is 'skyblue'.
    -figsize (tuple): Size of the figure. Default is (8, 5).

    Returns:
    -None
    """
    # Set up the figure
    plt.figure(figsize=figsize)

    # Create the count plot
    ax = sns.countplot(data=df, x=target_col, color=color)

    # Annotate each bar with the count
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',  # Count value
                    (p.get_x() + p.get_width() / 2., p.get_height()),  # Positioning
                    ha='center', va='bottom',  # Center alignment
                    fontsize=12)  # Font size for annotations

    # Add titles and labels
    plt.title(f"{target_col} Distribution", fontsize=16)
    plt.xlabel(target_col, fontsize=14)
    plt.ylabel("Count", fontsize=14)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


# Call the function
plot_target_distribution(df)


def upsample_minority_class(df, target_col='Class', majority_class=0, minority_class=1, random_state=42, plot=True):
    """
    Upsamples the minority class in the DataFrame to balance with the majority class,
    combines both classes, shuffles, and optionally visualizes the class distribution.

    Parameters:
    -df (pd.DataFrame): The input DataFrame.
    -target_col (str): The name of the target column with classes. Default is 'Class'.
    -majority_class (int): The label for the majority class. Default is 0.
    -minority_class (int): The label for the minority class. Default is 1.
    -random_state (int): Random seed for reproducibility. Default is 42.
    -plot (bool): If True, displays a plot of the class distribution after upsampling.

    Returns:
    -pd.DataFrame: The upsampled and shuffled DataFrame.
    """
    # Step 1: Separate majority and minority classes
    df_majority = df[df[target_col] == majority_class]
    df_minority = df[df[target_col] == minority_class]

    # Step 2: Upsample the minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # Sample with replacement
                                     n_samples=len(df_majority),  # Match majority class size
                                     random_state=random_state)  # For reproducible results

    # Step 3: Combine the upsampled minority class with the majority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    # Step 4: Shuffle the combined DataFrame
    df_upsampled = df_upsampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Step 5: Display class distribution after upsampling
    print("\nClass distribution after upsampling:")
    print(df_upsampled[target_col].value_counts())

    # Step 6: plot the class distribution
    if plot:
        plt.figure(figsize=(8, 5))  # Set figure size
        sns.countplot(data=df_upsampled, x=target_col, color='skyblue')  # Plot class distribution
        plt.title(f"{target_col} Distribution after Upsampling", fontsize=16)
        plt.xlabel(target_col, fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.tight_layout()
        plt.show()

    return df_upsampled


# Call the function
df_upsampled = upsample_minority_class(df)

# Round the features to 4 decimal places (excluding the target column)
df_upsampled.iloc[:, :-1] = df_upsampled.iloc[:, :-1].round(4)  # Round all columns except the target column

# Save the cleaned and upsampled DataFrame to a CSV file
file_name = 'upsampled_data.csv'  # Specify the name of the file
df_upsampled.to_csv(file_name, index=False)  # Save the DataFrame

# save the upsampled data to Desktop
df_upsampled.to_csv(r'C:\Users\USER\Desktop\file_name', index=False)

# Get the full path of the saved file
full_path = os.path.abspath(file_name)  # Get the absolute path of the filename
print(f"The upsampled data has been saved to: {full_path}")


