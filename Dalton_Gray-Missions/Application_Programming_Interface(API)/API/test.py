import requests
# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://127.0.0.1:5000/predict"


# Test Images:

#IMAGE_PATH = "fake01.jpg"
IMAGE_PATH = "fake02.jpg"
#IMAGE_PATH = "fake03.jpg"

#IMAGE_PATH = "real01.jpg"
#IMAGE_PATH = "real02.jpg"
#IMAGE_PATH = "real03.jpg"


# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was successful
if r["success"]:
    # loop over the predictions and display them
    for (i, result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(i + 1, result["label"],
            result["probability"]))

# otherwise, the request failed
else:
    print("Request failed")