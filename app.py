# importing the required libraries

from flask import Flask, render_template, request
import librosa
import numpy as np
from keras.models import load_model

from werkzeug.utils import secure_filename
import pickle 
pkl_file = open('label_encoder.pkl', 'rb')
labelencoder = pickle.load(pkl_file) 
model = load_model('my_model.h5')

# initialising the flask app
app = Flask(__name__)


def predict_model(filename):
    # filename="Downloads/audio/Gunshot.wav"
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)#We have one rwo with 40 features here

    predicted_label=np.argmax(model.predict(mfccs_scaled_features), axis=-1)
    #Labels tells about the class audio belongs to...

    print(predicted_label)
    prediction_class = labelencoder.inverse_transform(predicted_label) 
    return prediction_class
    
@app.route('/')
def upload_file():
   return render_template('index.html')

   
#Handling error 404 and displaying relevant web page
@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html'),404
 
#Handling error 500 and displaying relevant web page
@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html'),500
@app.route('/upload', methods = ['GET', 'POST'])
def uploadfile():
   if request.method == 'POST': # check if the method is post
      f = request.files['file'] # get the file from the files object
      f.save(secure_filename(f.filename)) # this will secure the file
      print(f.filename)
      output = predict_model(f.filename)
      print(output)
      #return 'file uploaded successfully' + str(output) # Display this message after uploading
      return render_template('upload.html',variable=output)

if __name__ == '__main__':
	app.run()


