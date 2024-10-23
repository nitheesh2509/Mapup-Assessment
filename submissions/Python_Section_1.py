from typing import Dict, List, Any
import pandas as pd
import numpy as np
import re
from itertools import permutations
from datetime import datetime, timedelta

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    return [elem for i in range(0, len(lst), n) for elem in reversed(lst[i:i+n])]

lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
n = 3
print(reverse_by_n_elements(lst, n)) 

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Initialize an empty dictionary to store the result
    length_dict = {}
    
    # Traverse the list and group strings by their length
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    
    # Return the dictionary sorted by keys (string lengths)
    return dict(sorted(length_dict.items()))

# Example usage:
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))


def flatten_dict(nested_dict: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys,
    including handling of lists by adding list indexing to the keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    flat_dict = {}

    def flatten(current_dict, parent_key=''):
        for k, v in current_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                flatten(v, new_key)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    list_key = f"{new_key}[{i}]"
                    if isinstance(item, dict):
                        flatten(item, list_key)
                    else:
                        flat_dict[list_key] = item
            else:
                flat_dict[new_key] = v

    flatten(nested_dict)
    return flat_dict

# Example usage:
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

flattened_dict = flatten_dict(nested_dict)
print(flattened_dict)

# Expected Output:
# {
#     "road.name": "Highway 1",
#     "road.length": 350,
#     "road.sections[0].id": 1,
#     "road.sections[0].condition.pavement": "good",
#     "road.sections[0].condition.traffic": "moderate"
# }

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Function to recursively generate all permutations
    def backtrack(start, end):
        if start == end:
            result.append(nums[:])  # Append a copy of the current permutation
        seen = set()  # Track the numbers we've already placed at the current position
        for i in range(start, end):
            if nums[i] not in seen:  # Only proceed if we haven't used this number at this position
                seen.add(nums[i])
                nums[start], nums[i] = nums[i], nums[start]  # Swap the current element with start
                backtrack(start + 1, end)  # Recurse to generate permutations for the next position
                nums[start], nums[i] = nums[i], nums[start]  # Backtrack (swap back to restore state)

    result = []
    nums.sort()  # Sort to ensure duplicates are adjacent
    backtrack(0, len(nums))
    return result

# Example usage:
nums = [1, 1, 2]
unique_perms = unique_permutations(nums)
print(unique_perms)

# Expected Output:
# [
#     [1, 1, 2],
#     [1, 2, 1],
#     [2, 1, 1]
# ]

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    # Define the regex patterns for different date formats
    date_patterns = [
        r'\b(\d{2})-(\d{2})-(\d{4})\b',      # dd-mm-yyyy
        r'\b(\d{2})/(\d{2})/(\d{4})\b',      # mm/dd/yyyy
        r'\b(\d{4})\.(\d{2})\.(\d{2})\b'     # yyyy.mm.dd
    ]
    dates = []
    
    # Find and format the matched dates
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if pattern == date_patterns[0]:  # dd-mm-yyyy
                dates.append(f"{match[0]}-{match[1]}-{match[2]}")
            elif pattern == date_patterns[1]:  # mm/dd/yyyy
                dates.append(f"{match[0]}/{match[1]}/{match[2]}")
            elif pattern == date_patterns[2]:  # yyyy.mm.dd
                dates.append(f"{match[0]}.{match[1]}.{match[2]}")
    
    return dates

# Example usage:
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
found_dates = find_all_dates(text)
print(found_dates)

# Expected Output:
# ["23-08-1994", "08/23/1994", "1994.08.23"]


import polyline
import math

def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in meters
    R = 6371000
    # Convert latitude and longitude from degrees to radians
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Distance in meters
    return R * c

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Step 1: Decode the polyline string into coordinates
    coords = polyline.decode(polyline_str)
    
    # Step 2: Create a DataFrame with the decoded coordinates
    df = pd.DataFrame(coords, columns=['latitude', 'longitude'])
    
    # Step 3: Calculate distances between consecutive points using the haversine formula
    distances = [0]  # First point has no previous point, so distance is 0
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude']
        lat2, lon2 = df.loc[i, 'latitude'], df.loc[i, 'longitude']
        dist = haversine(lat1, lon1, lat2, lon2)
        distances.append(dist)
    
    # Step 4: Add the distance column to the DataFrame
    df['distance'] = distances
    
    return df

# Example usage:
polyline_str = 'u{~vFvyys@fSxS'
df_result = polyline_to_dataframe(polyline_str)
print(df_result)

# Sample Output:
#    latitude  longitude     distance
# 0   38.5000  -120.2000       0.0000
# 1   40.7000  -120.9500  248585.7231
# 2   43.2520  -126.4530  595243.4798

from typing import List

def rotate_and_transform_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then replace each element 
    with the sum of all elements in its row and column (in the rotated matrix), excluding itself.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    if not matrix or not matrix[0]:
        return []
    
    n = len(matrix)
    
    # Step 1: Rotate the matrix 90 degrees clockwise
    rotated = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            rotated[j][n - 1 - i] = matrix[i][j]
    
    # Step 2: Calculate the sum of elements in each row and column of the rotated matrix
    row_sums = [0] * n
    col_sums = [0] * n
    
    for i in range(n):
        for j in range(n):
            row_sums[i] += rotated[i][j]
            col_sums[j] += rotated[i][j]
    
    # Step 3: Create the transformed matrix
    transformed = [[0] * n for _ in range(n)]  # Initialize the transformed matrix

    for i in range(n):
        for j in range(n):
            # Replace the current element with the row sum + column sum - itself
            transformed[i][j] = row_sums[i] + col_sums[j] - rotated[i][j]

    return transformed

# Example usage:
matrix = [[1, 2, 3], 
          [4, 5, 6], 
          [7, 8, 9]]
result = rotate_and_transform_matrix(matrix)
print(result)  # Expected output: [[22, 19, 16], [23, 20, 17], [24, 21, 18]]

def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Verify the completeness of the time data by checking whether the timestamps for each unique (`id`, `id_2`) pair
    cover a full 24-hour period and span all 7 days of the week.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'id', 'id_2', 'startDay', 'startTime', 'endDay', 'endTime'.

    Returns:
        pd.Series: A boolean series with multi-index (id, id_2) indicating if each pair has incorrect timestamps.
    """
    # Check if required columns are present
    required_columns = ['id', 'id_2', 'startDay', 'startTime', 'endDay', 'endTime']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("The DataFrame must contain the following columns: " + ", ".join(required_columns))

    # Create a base date for testing purposes
    base_date = '2024-01-01'

    # Standardize and create full datetime strings for start and end
    df['start_datetime_str'] = df['startDay'].str.strip() + ' ' + df['startTime'].str.strip()
    df['end_datetime_str'] = df['endDay'].str.strip() + ' ' + df['endTime'].str.strip()

    # Combine base date with the strings for proper parsing
    df['start_datetime'] = pd.to_datetime(base_date + ' ' + df['start_datetime_str'], format='%Y-%m-%d %A %H:%M:%S', errors='coerce')
    df['end_datetime'] = pd.to_datetime(base_date + ' ' + df['end_datetime_str'], format='%Y-%m-%d %A %H:%M:%S', errors='coerce')

    # Check for NaT values after conversion
    print("\nRows with NaT values in start_datetime or end_datetime:")
    print(df[df['start_datetime'].isna() | df['end_datetime'].isna()])

    # Drop rows where datetime conversion failed
    df.dropna(subset=['start_datetime', 'end_datetime'], inplace=True)

    # Show remaining DataFrame after dropping NaT values
    print("\nRemaining DataFrame after datetime conversion:")
    print(df)

    # Group by (id, id_2)
    grouped = df.groupby(['id', 'id_2'])

    def check_timestamps(group):
        # Check if there are timestamps for all 7 days of the week
        days_covered = group['start_datetime'].dt.day_name().unique()
        full_days = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'}
        
        # Check if all days are covered
        all_days_covered = full_days.issubset(days_covered)
        
        # Check if the timestamps cover a full 24-hour period
        full_24_hours = (group['start_datetime'].min() <= group['end_datetime'].max() - pd.Timedelta(hours=24))
        
        # Return True if timestamps are incorrect (not covering all days and 24 hours)
        return not (all_days_covered and full_24_hours)

    # Apply the check function to each group and create a boolean series
    result = grouped.apply(check_timestamps)

    # Return a boolean series with multi-index (id, id_2)
    return result

# Example usage:
if __name__ == "__main__":
    # Path to the dataset-1.csv file
    dataset_path = r'E:\Mapup\Mapup-Assessment\datasets\dataset-1.csv'

    # Read the dataset
    df = pd.read_csv(dataset_path)

    # Run the time_check function
    incorrect_timestamps = time_check(df)

    # Print the results
    print("\nIncorrect Timestamps:")
    print(incorrect_timestamps[incorrect_timestamps])  # Only print True values
