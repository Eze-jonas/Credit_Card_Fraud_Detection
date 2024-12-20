# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 19:53:41 2024

@author: USER
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import yeojohnson
from scipy.stats.mstats import winsorize as winsorize_func  
import os

# Import the dataset.
# specify path to the dataset
def load_dataset(file_path, expected_columns=None):
    """
    Load a dataset from a specified file path and perform checks.

    Parameters:
    - file_path: str, path to the dataset file.
    - expected_columns: list, optional list of expected column names.

    Returns:
    - df: DataFrame containing the loaded dataset.

    Raises:
    - FileNotFoundError: If the file does not exist.
    - ValueError: If the file is empty or missing expected columns.
    - RuntimeError: For any other errors during loading.
    """
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} was not found. Please check the path and try again.")

    # Load dataset with error handling
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty. Please provide a valid dataset.")
    except pd.errors.ParserError:
        raise ValueError("There was an error parsing the file. Check if the file is in a valid CSV format.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the file: {e}")

    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("The DataFrame is empty. The file may not contain data or could be corrupted.")

    # Verify expected columns if provided
    if expected_columns:
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The following expected columns are missing: {missing_columns}")

    # Display few rows of the DataFrame to confirm it's loaded correctly
    print("DataFrame loaded successfully. Here are the first few rows:")
    print(df.head())

    return df

# Set file path and call the function outside the function body
file_path = r"C:\Users\USER\Desktop\credit_card_fraud_detection\creditcard.csv"
df = load_dataset(file_path)
# Examine number of PCS that can produce 90% information about the original data. 
def plot_pca_variance(df, threshold=0.90):
    """
    Plot the Scree Plot and Cumulative Explained Variance from PCA results in the DataFrame.
    Determine the number of principal components needed to reach a specified variance threshold.

    Parameters:
    - df: DataFrame containing the PCA results with columns V1 to V28.
    - threshold: float, default=0.90, the variance threshold to determine the number of principal components to retain.

    Returns:
    - num_pcs_to_keep: int, number of principal components required to reach the threshold variance.
    """
    
    # Precaution 1: Check if DataFrame is valid
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None. Please provide a valid DataFrame.")

    # Precaution 2: Check for required columns in DataFrame
    expected_columns = [f'V{i}' for i in range(1, 29)]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in DataFrame: {missing_columns}")

    # Precaution 3: Check that the columns are numeric
    if not all(np.issubdtype(df[col].dtype, np.number) for col in expected_columns):
        raise ValueError("All columns from V1 to V28 must be numeric.")

    # Extract the principal components from the DataFrame
    pcs = df[expected_columns]

    # Precaution 4: Handle errors in PCA fitting
    try:
        # Initialize PCA to determine explained variance
        pca = PCA()
        pca.fit(pcs)
    except Exception as e:
        raise RuntimeError(f"PCA fitting failed. Check data for issues. Error: {e}")

    # Explained variance and cumulative variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    # Plotting the Scree Plot and Cumulative Explained Variance
    plt.figure(figsize=(10, 5))

    # Scree Plot
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')

    # Cumulative Explained Variance Plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'{threshold*100}% Variance')
    plt.xlabel('Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.xticks(range(1, len(cumulative_variance) + 1))
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Determine the number of PCs to retain for the specified threshold
    num_pcs_to_keep = (cumulative_variance >= threshold).argmax() + 1  # +1 for index

    print(f'Number of PCs to keep for {threshold*100}% variance: {num_pcs_to_keep}')
    
    return num_pcs_to_keep
num_pcs_to_keep = plot_pca_variance(df)
# Drop extra pcs

def drop_columns(df, columns):
    """
    Drop specified columns from a DataFrame.

    Parameters:
    - df: Pandas DataFrame from which to drop columns.
    - columns: List of column names to drop (list of strings).

    Returns:
    - None: The function modifies the input DataFrame in place.
    
    Raises:
    - TypeError: If df is not a DataFrame or columns is not a list of strings.
    - ValueError: If any specified column is missing in the DataFrame.
    """
    # Precaution 1: Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The first parameter must be a pandas DataFrame.")

    # Precaution 2: Check if columns is a list
    if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
        raise TypeError("The columns parameter must be a list of strings representing column names to drop.")
    
    # Precaution 3: Check if columns exist in the DataFrame
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The following columns are missing from the DataFrame and cannot be dropped: {missing_columns}")
    
    # Drop the columns
    df.drop(columns=columns, inplace=True)
columns_to_drop = ['V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
drop_columns(df, columns_to_drop)

# Display the modified DataFrame
print(df.head())

# Examine total rows of the dataset.
def display_total_rows(df):
    """
    Display the total number of rows in the dataset.

    Parameters:
    - df: DataFrame for which the total row count is to be displayed.

    Returns:
    - int: Total number of rows in the DataFrame.

    Raises:
    - TypeError: If df is not a pandas DataFrame.
    - ValueError: If df is empty.
    """
    # Precaution 1: Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input must be a pandas DataFrame.")

    # Precaution 2: Check if the DataFrame is empty
    if df.empty:
        raise ValueError("The DataFrame is empty. No rows to display.")

    # Calculate and display the total number of rows
    total_rows = len(df)
    print(f"Total rows of the dataset: {total_rows}")
    return total_rows

# Call the function
total_rows = display_total_rows(df)

# Examine the data type of the dataset.
def display_data_types(df):
    """
    Display the data types of each column in the DataFrame.
    
    Parameters:
    - df: DataFrame for which the column data types are to be displayed.
    
    Returns:
    - Series: Data types of each column in the DataFrame.
    
    Raises:
    - TypeError: If df is not a DataFrame.
    - ValueError: If the DataFrame is empty.
    """
    # Precaution 1: Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input must be a pandas DataFrame.")
    
    # Precaution 2: Check if the DataFrame is empty
    if df.empty:
        raise ValueError("The DataFrame is empty. No data types to display.")

    data_types = df.dtypes
    print("These are the data types of the variables:")
    print(data_types)
    return data_types
# Call the function
data_type = display_data_types(df)

# Examines the dataset for a complete empty rows.
def check_completely_empty_rows(df):
    """
    Check if there are completely empty rows in the DataFrame and display the count.

    Parameters:
    - df: DataFrame to be checked for completely empty rows.

    Returns:
    - int: Number of completely empty rows in the DataFrame.

    Raises:
    - TypeError: If df is not a DataFrame.
    - ValueError: If the DataFrame is empty.
    """
    # Precaution 1: Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input must be a pandas DataFrame.")
    
    # Precaution 2: Check if the DataFrame is empty
    if df.empty:
        raise ValueError("The DataFrame is empty. There are no rows to check.")
    
    # Identify completely empty rows
    completely_empty_rows = df[df.isnull().all(axis=1)]
    
    # Display the result
    if not completely_empty_rows.empty:
        print(f"Number of completely empty rows: {len(completely_empty_rows)}")
    else:
        print("No row is completely empty")
    
    return len(completely_empty_rows)
# Call the function
completely_empty_rows = check_completely_empty_rows(df)

def check_duplicated_rows(df):
    """
    Check if there are duplicated rows in the DataFrame and display the count.

    Parameters:
    - df: DataFrame to be checked for duplicated rows.

    Returns:
    - int: Number of dulicated rows in the DataFrame.
    """
    # Identify duplicated rows
    duplicated_rows = df[df.duplicated( )]
    if not duplicated_rows.empty:
        print(f"Number of duplicated rows:{len(duplicated_rows)}")
        print("Duplicated rows are:")
        print(duplicated_rows)
    else:
        print("There is no duplicate rows:")
    return len(duplicated_rows)     
# Call the function
duplicated_rows = check_duplicated_rows(df)

# Drop the duplicated rows.
def drop_duplicated_rows(df):
    """
    Drops duplicated rows from the DataFrame to remove redundant data and displays the number of rows after removal.

    Parameters:
    - df: DataFrame from which duplicated rows will be dropped.

    Returns:
    - int: The length of the DataFrame after dropping duplicates.

    Raises:
    - TypeError: If df is not a DataFrame.
    - ValueError: If the DataFrame is empty.
    """
    # Precaution 1: Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input must be a pandas DataFrame.")
    
    # Precaution 2: Check if the DataFrame is empty
    if df.empty:
        raise ValueError("The DataFrame is empty. There are no rows to drop duplicates from.")
    
    # Drop duplicated rows
    original_len = len(df)
    df.drop_duplicates(inplace=True)
    new_len = len(df)
    
    # Display the number of rows before and after dropping duplicates
    print(f"Original number of rows: {original_len}")
    print(f"Number of rows after dropping duplicates: {new_len}")
    
    return new_len
# Call the function
new_len = drop_duplicated_rows(df)

# Examines the dataset for missing values.
def check_missing_values(df):
    """
    Checks missing values in the DataFrame.

    Parameters:
    - df: DataFrame from which missing values will be checked.

    Returns:
    - Series: Variables with count of missing values.

    Raises:
    - TypeError: If df is not a DataFrame.
    - ValueError: If the DataFrame is empty.
    """
    # Precaution 1: Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input must be a pandas DataFrame.")
    
    # Precaution 2: Check if the DataFrame is empty
    if df.empty:
        raise ValueError("The DataFrame is empty. There are no variables to check for missing values.")
    
    # Identify missing values (NaN)
    missing_values = df.isna().sum()
    print("These are the variables and their count of missing values:")
    print(missing_values)
    
    return missing_values
# Call the function
missing_value = check_missing_values(df)

# Transform, handle outliers and normalize the dataset
def handle_column(df, col_name, transformation=None, winsorize=False, limits=(0.05, 0.05), normalize=None):
    """
    Apply transformations to a DataFrame column in place, including outlier handling, transformations, and normalization.

    Parameters:
    - df: Pandas DataFrame containing the column to transform.
    - col_name: Name of the column to transform (string).
    - transformation: Type of transformation ('yeo_johnson').
    - winsorize: Boolean, if True applies winsorization to limit extreme values.
    - limits: Tuple, limits for winsorization. Default is (0.05, 0.05).
    - normalize: Type of normalization ('standard' or 'robust').

    Returns:
    - None: The function modifies the input DataFrame in place.
    """
    # Precaution 1: Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input must be a pandas DataFrame.")
    
    # Precaution 2: Check if the specified column exists in the DataFrame
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' does not exist in the DataFrame.")
    
    # Precaution 3: Check if the specified column is numeric
    if not pd.api.types.is_numeric_dtype(df[col_name]):
        raise TypeError(f"The column '{col_name}' must be numeric.")
    
    # Precaution 4: Validate limits for winsorization
    if not (0 <= limits[0] < 1 and 0 <= limits[1] < 1):
        raise ValueError("Limits for winsorization must be between 0 and 1.")
    col = df[col_name]  # Get the column from the DataFrame
    
    # Step 1: Apply Yeo-Johnson transformation if specified
    if transformation == 'yeo_johnson':
        df.loc[:, col_name] = yeojohnson(col)[0]  # Apply Yeo-Johnson transformation in place
    
    # Step 2: Apply Winsorization if specified
    if winsorize:
        df.loc[:, col_name] = winsorize_func(col, limits=limits)  # Apply winsorization in place
    
    # Step 3: Apply normalization if specified
    if normalize == 'standard':
        scaler = StandardScaler()
        df.loc[:, col_name] = scaler.fit_transform(col.values.reshape(-1, 1)).flatten()  # Apply standardization in place
    elif normalize == 'robust':
        scaler = RobustScaler()
        df.loc[:, col_name] = scaler.fit_transform(col.values.reshape(-1, 1)).flatten()  # Apply robust scaling in place

# Apply preprocessing to 'Time' and other specified columns
handle_column(df, 'Time', normalize='standard')
handle_column(df, 'V1', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='standard')
handle_column(df, 'V2', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='standard')
handle_column(df, 'V3', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='standard')
handle_column(df, 'V4', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='standard')
handle_column(df, 'V5', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='standard')
handle_column(df, 'V6', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='robust')
handle_column(df, 'V7', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='standard')
handle_column(df, 'V8', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='robust')
handle_column(df, 'V9', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='standard')
handle_column(df, 'V10', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='standard')
handle_column(df, 'V11', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='standard')
handle_column(df, 'V12', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='standard')
handle_column(df, 'V13', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='standard')
handle_column(df, 'V14', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='standard')
handle_column(df, 'V15', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='standard')
handle_column(df, 'V16', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='standard')
handle_column(df, 'V17', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='standard')
handle_column(df, 'V18', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='standard')
handle_column(df, 'V19', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='standard')
handle_column(df, 'V20', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='robust')
handle_column(df, 'Amount', transformation='yeo_johnson', winsorize=True, limits=(0.05, 0.05), normalize='standard')


# Save the cleaned and upsampled DataFrame to a CSV file
file_name = 'cleaned_data.csv'  # Specify the name of the file
df.to_csv(file_name, index=False)  # Save the DataFrame

# Get the full path of the saved file
full_path = os.path.abspath(file_name)  # Get the absolute path of the filename
print(f"The cleaned data has been saved to: {full_path}")


