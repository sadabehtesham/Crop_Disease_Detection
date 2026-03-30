from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import json
import uuid
import tensorflow as tf
import requests

app = Flask(__name__)

model = tf.keras.models.load_model("models/crop_disease_model.keras")

with open("plant_disease.json", 'r') as file:
    plant_disease = json.load(file)

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


def extract_features(image):
    image = tf.keras.utils.load_img(image, target_size=(160,160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature


def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label


@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        image.save(f'{temp_name}_{image.filename}')
        prediction = model_predict(f'./{temp_name}_{image.filename}')
        return render_template('home.html', result=True,
                               imagepath=f'/{temp_name}_{image.filename}',
                               prediction=prediction)
    else:
        return redirect('/')


@app.route('/weather')
def weather():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not lat or not lon:
        return {"error": True, "message": "Missing latitude or longitude"}, 400

    API_KEY = "bfd851a6ce85aad45ccc612b9321aecf"
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

    response = requests.get(url)
    data = response.json()

    if "main" not in data:
        return {
            "error": True,
            "message": data.get("message", "Weather API failed"),
            "raw": data
        }, 500

    return {
        "error": False,
        "location": data.get("name", "Unknown"),
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "wind": data["wind"]["speed"],
        "description": data["weather"][0]["description"]
    }

if __name__ == "__main__":
    app.run(debug=True)
