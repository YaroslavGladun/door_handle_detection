import os

dataset_path = "generated_dataset"

all_hash = set()
good_hash = set()

for file in os.listdir(dataset_path):
    if file.endswith(".jpg") and file[:3] == "rgb":
        all_hash.add(file.split('_')[1].split('.')[0])
    elif file.endswith(".jpg") and file[:13] == "visualization":
        good_hash.add(file.split('_')[1].split('.')[0])

to_remove = all_hash - good_hash

for file in os.listdir(dataset_path):
    if file.endswith(".jpg") and file[:3] == "rgb":
        hash = file.split('_')[1].split('.')[0]
        if hash in to_remove:
            os.remove(f"{dataset_path}/{file}")
            os.remove(f"{dataset_path}/depth_{hash}.npy")
            os.remove(f"{dataset_path}/box_{hash}.npy")
            os.remove(f"{dataset_path}/plane_{hash}.npy")
