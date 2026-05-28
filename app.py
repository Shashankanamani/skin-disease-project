from flask import Flask, request, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# LOAD MODEL
model = tf.keras.models.load_model("model.h5")

# CLASS NAMES
classes = [
    "atopic",
    "basal_cell_carcinoma",
    "melanoma",
    "psoriasis"
]

HTML = """

<h2>Skin Disease Detection</h2>

<form method="POST" enctype="multipart/form-data">

    <input type="file" name="file">

    <input type="submit">

</form>

{% if prediction %}

<h3>Prediction: {{ prediction }}</h3>

{% endif %}

"""

@app.route("/", methods=["GET", "POST"])

def predict():

    prediction = ""

    if request.method == "POST":

        file = request.files["file"]

        image = Image.open(file)

        image = image.resize((128,128))

        image = np.array(image) / 255.0

        image = np.expand_dims(image, axis=0)

        pred = model.predict(image)

        predicted_class = np.argmax(pred)

        prediction = classes[predicted_class]

    return render_template_string(
        HTML,
        prediction=prediction
    )

if __name__ == "__main__":

    app.run(debug=True)