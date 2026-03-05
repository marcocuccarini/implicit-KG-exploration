import csv
import random

def shuffle_csv(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
    
    # Shuffle the list in place
    random.shuffle(reader)
    
    # Save the shuffled data
    if reader:
        keys = reader[0].keys()
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(reader)

# Usage
shuffle_csv('dataset_split_test.csv', 'shuffled_dataset.csv')