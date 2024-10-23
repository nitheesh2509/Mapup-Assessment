import pandas as pd
import numpy as np

from datetime import time, timedelta

def calculate_distance_matrix(file_path: str) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the data from a CSV file containing 'id_start', 'id_end', and 'distance'.

    Args:
        file_path (str): Path to the CSV file containing 'id_start', 'id_end', 'distance'.

    Returns:
        pandas.DataFrame: Distance matrix.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Print the DataFrame columns to check their names
    print("Columns in the DataFrame:", df.columns)

    # Check the first few rows to understand the data structure
    print("First few rows of the DataFrame:\n", df.head())

    # Use the correct column names
    id_from_col = 'id_start'
    id_to_col = 'id_end'
    distance_col = 'distance'

    # Get unique IDs from both columns
    ids = pd.concat([df[id_from_col], df[id_to_col]]).unique()
    
    # Initialize the distance matrix with infinity
    distance_matrix = pd.DataFrame(np.inf, index=ids, columns=ids)
    np.fill_diagonal(distance_matrix.values, 0)  # Set diagonal to 0

    # Populate the distance matrix with known distances
    for _, row in df.iterrows():
        id_from = row[id_from_col]
        id_to = row[id_to_col]
        distance = row[distance_col]
        
        # Update the distance matrix with the minimum distance
        distance_matrix.loc[id_from, id_to] = min(distance_matrix.loc[id_from, id_to], distance)
        distance_matrix.loc[id_to, id_from] = min(distance_matrix.loc[id_to, id_from], distance)  # Ensure symmetry

    # Apply the Floyd-Warshall algorithm for all pairs shortest path
    for k in ids:
        for i in ids:
            for j in ids:
                if distance_matrix.loc[i, k] < np.inf and distance_matrix.loc[k, j] < np.inf:
                    distance_matrix.loc[i, j] = min(distance_matrix.loc[i, j], distance_matrix.loc[i, k] + distance_matrix.loc[k, j])
    
    return distance_matrix

# Example usage
distance_matrix = calculate_distance_matrix('dataset-2.csv')
print(distance_matrix)

def unroll_distance_matrix(distance_matrix: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        distance_matrix (pandas.DataFrame): Distance matrix.
        original_df (pandas.DataFrame): Original DataFrame to merge vehicle types.

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', 'distance', and 'vehicle_type'.
    """
    # Create an empty list to store the rows
    unrolled_data = []
    
    # Iterate through the distance matrix
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            # Skip self-distances (same id_start and id_end)
            if id_start != id_end:
                distance = distance_matrix.loc[id_start, id_end]
                # Only add to the unrolled data if distance is not infinite (meaning there's no connection)
                if distance < np.inf:
                    unrolled_data.append((id_start, id_end, distance))
    
    # Create a DataFrame from the list of tuples
    result_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])

    # Check the original DataFrame columns
    print("Columns in the original DataFrame:", original_df.columns)

    # Merge vehicle type information back from the original DataFrame
    # Ensure that the correct column names are used for merging
    if 'id' in original_df.columns and 'vehicle_type' in original_df.columns:
        result_df = result_df.merge(original_df[['id', 'vehicle_type']], left_on='id_start', right_on='id', how='left')
        result_df.drop('id', axis=1, inplace=True)
    else:
        print("Warning: 'id' and 'vehicle_type' columns not found in the original DataFrame.")

    return result_df

# Example usage
file_path = 'dataset-2.csv'  # Replace with your CSV file path
original_df = pd.read_csv(file_path)  # Load original DataFrame to get vehicle types
print("Original DataFrame head:\n", original_df.head())  # Check the first few rows of the original DataFrame

# Calculate the distance matrix
result_matrix = calculate_distance_matrix(file_path)

# Unroll the distance matrix into a DataFrame
result_df = unroll_distance_matrix(result_matrix, original_df)

# Display the result DataFrame
print("Unrolled distance DataFrame:")
print(result_df.head(10))


# Print the unrolled DataFrame
print("Unrolled distance DataFrame with vehicle types:")
print(result_df)


# Sample DataFrame based on your previous description
data = {
    'id_start': [1001400, 1001402, 1001404, 1001406, 1001408, 1001410],
    'id_end': [1001402, 1001404, 1001406, 1001408, 1001410, 1001412],
    'distance': [9.7, 20.2, 16.0, 21.7, 11.1, 15.6]
}

df = pd.DataFrame(data)

def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): DataFrame with distances.
        reference_id (int): The reference ID.

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold.
    """
    # Calculate the average distance for the reference ID
    average_distance = df[df['id_start'] == reference_id]['distance'].mean()

    # Check if average_distance is NaN (no entries for the reference_id)
    if pd.isna(average_distance):
        return pd.DataFrame(columns=['id_start', 'distance'])  # Return empty DataFrame if no distances found

    # Calculate the lower and upper bounds
    lower_bound = average_distance * 0.9
    upper_bound = average_distance * 1.1

    # Find similar IDs within the specified range
    similar_ids = df[(df['id_start'] != reference_id) & 
                     (df['distance'] >= lower_bound) & 
                     (df['distance'] <= upper_bound)]

    # Return a DataFrame with unique IDs and their distances
    return similar_ids[['id_start', 'distance']].drop_duplicates()

# Example usage
reference_id = 1001402
result_df = find_ids_within_ten_percentage_threshold(df, reference_id)
print("IDs within 10% of the average distance:")
print(result_df)


def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the distance.

    Args:
        df (pandas.DataFrame): DataFrame with 'distance' and 'vehicle_type'.

    Returns:
        pandas.DataFrame: DataFrame with calculated toll rates for each vehicle type.
    """
    # Define the rate coefficients for each vehicle type
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate the toll rates by multiplying the distance with the rate coefficients
    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate

    return df

# Example usage with the provided distances
data = {
    'id_start': [1001400] * 10,  # Constant id_start
    'id_end': [
        1001402, 1001404, 1001406, 1001408, 1001410, 
        1001412, 1001414, 1001416, 1001418, 1001420
    ],  # Different id_end values
    'distance': [
        9.7, 29.9, 45.9, 67.6, 78.7, 
        94.3, 112.5, 125.7, 139.3, 152.2
    ]  # Provided distances
}

df = pd.DataFrame(data)

# Calculate toll rates
result_df = calculate_toll_rate(df)

# Select the required columns for the final output
result_df = result_df[['id_start', 'id_end', 'moto', 'car', 'rv', 'bus', 'truck']]

# Display the resulting DataFrame
print(result_df)

def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame): DataFrame with distance and vehicle toll rates.

    Returns:
        pandas.DataFrame: DataFrame with time-based toll rates.
    """
    # Define start and end times for the 24-hour period
    start_time = time(0, 0)  # 00:00:00
    end_time = time(23, 59, 59)  # 23:59:59

    # Create a list of days and time ranges
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Prepare lists to collect rows
    rows = []

    # Calculate toll rates for each unique (id_start, id_end) pair for each day
    for _, row in df.iterrows():
        for day in days_of_week:
            # Assign start and end times for the day
            current_start_time = start_time
            current_end_time = end_time

            # Apply discount based on the day of the week
            if day in days_of_week[:5]:  # Weekdays
                # Define the time intervals and multipliers
                time_intervals = [
                    (time(0, 0), time(10, 0), 0.8),   # Early Morning
                    (time(10, 0), time(18, 0), 1.2),  # Day
                    (time(18, 0), time(23, 59), 0.8)   # Evening
                ]
            else:  # Weekends
                time_intervals = [(time(0, 0), time(23, 59), 0.7)]  # Constant discount for all times

            # Calculate toll for each time interval
            for start, end, multiplier in time_intervals:
                rows.append({
                    'id_start': row['id_start'],
                    'id_end': row['id_end'],
                    'distance': row['distance'],
                    'moto': row['moto'] * multiplier,
                    'car': row['car'] * multiplier,
                    'rv': row['rv'] * multiplier,
                    'bus': row['bus'] * multiplier,
                    'truck': row['truck'] * multiplier,
                    'start_day': day,
                    'start_time': start,
                    'end_day': day,
                    'end_time': end
                })

    # Create a new DataFrame from the collected rows
    result_df = pd.DataFrame(rows)

    return result_df

# Example DataFrame with vehicle tolls
data = {
    'id_start': [1001400] * 10,
    'id_end': [
        1001402, 1001404, 1001406, 1001408, 1001410, 
        1001412, 1001414, 1001416, 1001418, 1001420
    ],
    'distance': [9.7, 29.9, 45.9, 67.6, 78.7, 94.3, 112.5, 125.7, 139.3, 152.2],
    'moto': [7.76, 23.92, 36.72, 54.08, 62.96, 75.44, 90.00, 100.56, 111.44, 121.76],
    'car': [11.64, 35.88, 55.08, 81.12, 94.44, 113.12, 135.00, 150.84, 167.16, 182.64],
    'rv': [14.55, 44.85, 68.85, 101.40, 117.90, 141.45, 168.75, 188.55, 208.95, 227.55],
    'bus': [21.34, 65.78, 100.98, 148.72, 173.04, 207.46, 247.50, 276.54, 305.88, 332.86],
    'truck': [34.92, 107.64, 165.24, 242.16, 282.92, 337.80, 405.00, 451.32, 490.92, 553.92],
}

df = pd.DataFrame(data)

# Calculate time-based toll rates
result_df = calculate_time_based_toll_rates(df)

# Display the resulting DataFrame
print(result_df)
