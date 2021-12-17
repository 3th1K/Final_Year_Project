print("Importing Libraries ...")
from flask import Flask, request, render_template
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json
import librosa
from librosa.feature.spectral import mfcc
import numpy as np
from pydub import AudioSegment
import os

# loading and Compiling the model
print("Compiling Model ... ")
json_file = open('ModelData/model.json', 'r')
loaded_json_model = json_file.read()
json_file.close()
model = model_from_json(loaded_json_model)
model.load_weights('ModelData/model.h5')
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")

# Initializing Flask
app = Flask(__name__)

# For the Home Page
@app.route('/')
def home():
    return render_template('index.html')

# After Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']

        # Check only for 'wav' and 'mp3'
        if file and (str(file.filename).rsplit('.',1)[1] == "wav" or str(file.filename).rsplit('.',1)[1] == "mp3"):
            
            # If given 'mp3' convert to 'wav'
            if str(file.filename).rsplit('.',1)[1] == "mp3":
                print("Converting mp3 to wav")
                sound = AudioSegment.from_mp3(file)
                sound.export("converted_to_wav.wav", format="wav")
                file = "converted_to_wav.wav"
                print("Converted")

            # Load the music file and extract features to analyze
            audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
            if os.path.exists("converted_to_wav.wav"):
                os.remove("converted_to_wav.wav") # deleting unwanted converted file
            mfcc_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfcc_scaled_features = np.mean(mfcc_features.T, axis=0)

            # Scaling features to be used in prediction
            music_metadata = mfcc_scaled_features.reshape(1, -1)

            # Prediction
            total_prediction = model.predict(music_metadata)

            # Trimming Prediction
            genre_prediction_as_index = np.argmax(total_prediction, axis=1)[0]

            # Distinguish Prediction as Genre
            dic = {0:"Blues", 1:"Classical", 2:"Country", 3:"Disco", 4:"Hiphop", 5:"Jazz", 6:"Metal", 7:"Pop", 8:"Reggae", 9:"Rock"}
            if genre_prediction_as_index in dic.keys():
                predicted_genre = dic[genre_prediction_as_index]
            else:
                predicted_genre = "Cannot Predict Genre For This Music"

            # output sent to frontend
            return render_template('index.html', prediction_text = predicted_genre)
        return render_template('index.html', prediction_text="Please Select A Music File")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1234, debug=True)
