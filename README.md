# Image Classification Model
This is a image classification model using inception model from tensorflow. This model is written in python 2.7.

### Installation
1. Install python 2.7
2. Install pip
3. Run command - pip install -r requirements.txt

### Train the model.
Run command - python train.py

### Classify Test set
Run command - python classify.py <image_path>

### Image Augmentation
To create more variation in the dataset using the given train test, use the data_augmentation.py script. It will take the data from train set and apply multiple algorithms to generate variation in the datasets. The resulting images will be stored in the images folder under the appropriate labels. 
Run command - python data_augmentation.py
