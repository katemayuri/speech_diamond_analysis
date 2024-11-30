#code to combine and load dataset into json file
import os
import json

# Define your folders
folder1 = "/mnt/c/Users/mayur/Desktop/Dal - Fall 2024/CSCI 6518 - Deep Speech/Project/Data Set Analysis/Data-Set/Audio/Rapaport Podcasts/label_data"
folder2 = "/mnt/c/Users/mayur/Desktop/Dal - Fall 2024/CSCI 6518 - Deep Speech/Project/Data Set Analysis/Data-Set/Audio/Youtube market Analysis/label_data"
combined_data = []

# Combine files from both folders
for folder in [folder1, folder2]:
    for file_name in os.listdir(folder):
        if file_name.endswith(".json"):  # Assuming JSON format
            with open(os.path.join(folder, file_name), 'r') as f:
                data = json.load(f)
                combined_data.extend(data)  # Append data

# Save the combined data into a single file
with open("combined_dataset.json", 'w') as f:
    json.dump(combined_data, f, indent=4)

print("Combined dataset saved as 'combined_dataset.json'")
