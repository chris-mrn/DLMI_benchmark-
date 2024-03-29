import pandas as pd
import os
import numpy as np
from PIL import Image


def load_images_from_folder(folder_path):

    images = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Check if the file is a regular file and has a JPEG extension
        if os.path.isfile(file_path) and file_path.lower().endswith('.jpg'):
            # Open the image using PIL's Image.open
            with Image.open(file_path) as image:
                # Convert the image to numpy array and append to the list
                width, height = image.size
                left = width/4
                top = height/4
                right = 3*width/4
                bottom = 3*height/4
                # Crop the center of the image
                img_crop = image.crop((left, top, right, bottom))
                img = np.array(img_crop)
                images.append(img)

    return images


def load_data(csv_path, images_path):
    # Load the CSV file using pandas
    dataframe = pd.read_csv(csv_path)

    # Create a dictionary to store the information
    data_dict = {}

    # Iterate through the rows of the DataFrame .
    # and add the informationto the dictionary
    for index, row in dataframe.iterrows():
        # Make sure to adjust the column names
        # based on your actual CSV structure
        label = row['LABEL']
        date_of_birth = row['DOB']
        gender = row['GENDER']
        identifier = row['ID']
        lymphocyte_count = row['LYMPH_COUNT']
        img_path = images_path + identifier
        images = load_images_from_folder(img_path)

        # Add the information to the dictionary
        data_dict[index] = {
            'label': label,
            'dob': date_of_birth,
            'gender': gender,
            'id': identifier,
            'lymph_count': lymphocyte_count,
            'images': images
        }

    return data_dict


def load_X_y(data):
    X = []
    y = []
    for i in range(len(data)):
        X.append(data[i]['images'])
        y.append([data[i]['label']])
        print("loading images from patient:", i)
    return X, y


def load_data_bio(data):
    X = []
    y = []
    for i in range(len(data)):
        dob = data[i]['dob']
        age = float(dob.split('/')[2] if '/' in dob else dob.split('-')[2])
        sex = data[i]['gender']
        if sex == 'M':
            sex = 1
        else:
            sex = -1

        X.append([data[i]['lymph_count'], age, sex])
        y.append(data[i]['label'])
    return np.array(X), np.array(y)
