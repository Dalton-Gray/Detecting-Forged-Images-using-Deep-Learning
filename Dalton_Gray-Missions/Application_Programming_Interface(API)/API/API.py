# import the necessary packages
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.applications import imagenet_utils
import numpy as np
from PIL import Image
import flask
from flask import jsonify
import io
import json
from tensorflow.python.keras.backend import set_session

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)
# previously saved model is loaded from .h5 file
model = load_model('model_FINAL.h5')
labels = ["Fake", "Real"]
def prepare_image(image, target):
    image = image.resize(target)
    img = np.asarray(image)
    img = np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
    return img



@app.route('/')
def root_response():
    return "Hello World."



@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            # set to dimensions used in training
            image = prepare_image(image, target=(300, 300))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            global sess
            global graph
            with graph.as_default():
                set_session(sess)
                preds = model.predict_proba(image)[0][0]
                print(preds)
                if(int(preds) == 0):
                    prob1 = 1.0
                    prob2 = 1 - prob1
                else: 
                    prob2 = 1.0
                    prob1 = 1 - prob2

            results = [[labels[0],prob1], [labels[1], prob2]]
            data["predictions"] = []



            # loop over the results and add them to the list of returned predictions
            for (label, prob) in results:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)



# if this is the main thread of execution first load the model and
# then start the server

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run()