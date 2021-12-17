print("Importing The Required Libraries ...")
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.optimizers import Adam


# set dataset paths
Data_genre = "Data/genres_original"
Data_features = "Data/features_30_sec.csv"

# read the csv dataset
print("Reading Dataset ...")
metadata = pd.read_csv(Data_features)

# function for extracting features from music
def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type="kaiser_test")
    mfcc_features = librosa.feature.mfcc(y = audio, sr = sample_rate, n_mfcc=40)
    mfcc_scaled_features = np.mean(mfcc_features.T, axis=0)
    return mfcc_scaled_features

extracted_features = [] # for storing every extracted features from the dataset

# extracting music dataset features using the csv dataset
print("Extracting Features ... ")
for index, row in tqdm(metadata.iterrows()):
    try:
        label = str(row["label"])
        name = str(row["filename"])
        filename = os.path.join(os.path.abspath(Data_genre), label+'\\',name)
        
        extracted_feature = features_extractor(filename)
        extracted_features.append([extracted_feature, label])
    except Exception as e:
        print(f"File Error {e}")
        continue

# making dataframe of the extracted features
extracted_features_dataframe = pd.DataFrame(extracted_features, columns=["feature", "class"])

# dividing both parts of the dataframe
print("Splitting DataFrame ...")
x = np.array(extracted_features_dataframe["feature"].tolist())
y = np.array(extracted_features_dataframe["class"].tolist())

# labeling "class"/"y" part as it is multinumbered
labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))

# splitting x and y into training and testing parts 80%-20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# making the model
print("Making Model ...")
model = Sequential()
model.add(Dense(1024, input_shape=(40,), activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(y.shape[1], activation="softmax"))

# compiling the model
print("Compiling Model ... ")
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")

# training the model
print("Training the Model ...")
number_of_epochs = 100
batch_size = 32
model.fit(x_train, y_train, epochs=number_of_epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=1)

# evaluating model
ev_result = model.evaluate(x_test, y_test, verbose=1)
print(ev_result)

# saving the model in file
print("Saving the Model ...")
model_json = model.to_json()
with open('ModelData/model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('ModelData/model.h5')

print("Saved The Model")