import os
from collections import Counter

def count_first_numbers(directory):
    freq_counter = Counter()

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                line = file.readline().strip()
                if line:
                    first_number = line.split()[0]
                    freq_counter[first_number] += 1

    return freq_counter

# Example usage:
directory_path = 'C:\coding\minor project\AI ENABLED SYSTEM FOR ROAD SIGN\yolov5_data\labels\\train'  # Replace this with your actual folder path
frequencies = count_first_numbers(directory_path)

# Print results
for number, freq in sorted(frequencies.items()):
    print(f"{number}: {freq}")
