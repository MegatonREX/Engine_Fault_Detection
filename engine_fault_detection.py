import os
import io
import librosa
import numpy as np
import warnings
import tensorflow as tf
from pydub import AudioSegment
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=UserWarning)

model = load_model("engine_fault_classifier.h5")

label_map = {0: "Faulty", 1: "Not Faulty"}

def features_extractor(file):
    audio, sample_rate = librosa.load(file, sr=None)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfcc_scaled_features

def predict_audio(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".wav":
        features = features_extractor(file_path)
    else:
        audio = AudioSegment.from_file(file_path)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        features = features_extractor(buffer)
    
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features, verbose=0) 
    
    predicted_class = int(prediction[0] > 0.5)
    print("Probability:", prediction[0])
    print("Predicted Class:", predicted_class)
    print("Predicted Label:", label_map[predicted_class])

file_path = r"Car-Engine-Sounds-Dataset-main\Abnormal Car Sounds/Ford Fiesta 2014 engine knocking.wav"
predict_audio(file_path)
