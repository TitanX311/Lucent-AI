import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

base_dir = "data/"
label_dir = "labels/"
image_dir = "images/"

label_format = ".json"
image_format = ".png"

train_label_dir = os.path.join(base_dir, "labels/train/")
val_label_dir = os.path.join(base_dir, "labels/val/")
train_image_dir = os.path.join(base_dir, "images/train/")
val_image_dir = os.path.join(base_dir, "images/val/")

os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)

labels = os.listdir(label_dir)

train_labels, val_labels = train_test_split(labels, test_size=0.1, random_state=92)

for label in tqdm(train_labels, desc="Copying train labels and images"):
    src = os.path.join(label_dir, label)
    dst = os.path.join(train_label_dir, label)
    shutil.copy(src, dst)

    # image copy
    src = os.path.join(image_dir, label.replace(label_format, image_format))
    dst = os.path.join(train_image_dir, label.replace(label_format, image_format))
    shutil.copy(src, dst)

for label in tqdm(val_labels, desc="Copying validation labels and images"):
    src = os.path.join(label_dir, label)
    dst = os.path.join(val_label_dir, label)
    shutil.copy(src, dst)

    #image copy
    src = os.path.join(image_dir, label.replace(label_format, image_format))
    dst = os.path.join(val_image_dir, label.replace(label_format, image_format))
    shutil.copy(src, dst)

print("Data split completed.")
print(f"Total labels: {len(labels)}")
print(f"Train labels: {len(train_labels)}, Val labels: {len(val_labels)}")
