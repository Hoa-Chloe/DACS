# import os
# import numpy as np
# import base64
# import io

# from flask import Flask, render_template, request
# from tensorflow import keras
# from PIL import Image

# # ======================
# # Đường dẫn project
# # ======================

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# app = Flask(
#     __name__,
#     template_folder=os.path.join(BASE_DIR, "templates"),
#     static_folder=os.path.join(BASE_DIR, "static")
# )

# # ======================
# # Config
# # ======================

# UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# MODEL_PATH = os.path.join(BASE_DIR, "skin_model.keras")

# IMG_SIZE = 224
# class_names = ['dry','normal','oily']

# # ======================
# # Load model
# # ======================

# print("Loading model...")
# model = keras.models.load_model(MODEL_PATH)
# print("Model loaded!")

# # ======================
# # Preprocess image
# # ======================

# def preprocess_image(img):

#     img = img.resize((IMG_SIZE,IMG_SIZE))
#     img = np.array(img)/255.0
#     img = np.expand_dims(img,axis=0)

#     return img


# # ======================
# # Predict
# # ======================

# def predict_image(img):

#     img_array = preprocess_image(img)

#     preds = model.predict(img_array)[0]

#     prediction = class_names[np.argmax(preds)]

#     probs = {
#         class_names[i]: round(float(preds[i])*100,2)
#         for i in range(len(class_names))
#     }

#     return prediction,probs


# # ======================
# # Route chính
# # ======================

# @app.route("/",methods=["GET","POST"])
# def index():

#     prediction=None
#     probs=None
#     image_path=None

#     if request.method=="POST":

#         # upload image
#         file=request.files.get("file")

#         if file and file.filename!="":

#             save_path=os.path.join(app.config["UPLOAD_FOLDER"],file.filename)
#             file.save(save_path)

#             img=Image.open(save_path).convert("RGB")

#             prediction,probs=predict_image(img)

#             image_path=os.path.join("static","uploads",file.filename)

#         # camera image
#         elif "image_data" in request.form:

#             image_data=request.form["image_data"]

#             image_data=image_data.split(",")[1]

#             image_bytes=base64.b64decode(image_data)

#             img=Image.open(io.BytesIO(image_bytes)).convert("RGB")

#             prediction,probs=predict_image(img)

#     return render_template(
#         "index.html",
#         prediction=prediction,
#         probs=probs,
#         image_path=image_path
#     )


# # ======================
# # Run server
# # ======================

# if __name__=="__main__":
#     app.run(debug=True)

import os
import numpy as np
import base64
import io

from flask import Flask, render_template, request
from tensorflow import keras
from PIL import Image

# ======================
# Đường dẫn project
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# ======================
# Config
# ======================
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# Tăng giới hạn dung lượng request lên 16MB để tránh lỗi Request Entity Too Large
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

MODEL_PATH = os.path.join(BASE_DIR, "skin_model.keras")

IMG_SIZE = 224
class_names = ['dry', 'normal', 'oily']

# ======================
# Load model
# ======================
print("Loading model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded!")
except Exception as e:
    print(f"Lỗi khi load model: {e}")
    model = None

# ======================
# Preprocess image
# ======================
def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ======================
# Predict
# ======================
def predict_image(img):
    if model is None:
        return "Model not loaded", {}
        
    img_array = preprocess_image(img)
    preds = model.predict(img_array)[0]
    prediction = class_names[np.argmax(preds)]

    probs = {
        class_names[i]: round(float(preds[i]) * 100, 2)
        for i in range(len(class_names))
    }
    return prediction, probs

# ======================
# Route chính
# ======================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probs = None
    image_path = None

    if request.method == "POST":
        # 1. Trường hợp upload ảnh từ máy
        file = request.files.get("file")
        if file and file.filename != "":
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(save_path)
            
            img = Image.open(save_path).convert("RGB")
            prediction, probs = predict_image(img)
            image_path = os.path.join("static", "uploads", file.filename)

        # 2. Trường hợp chụp ảnh từ camera (Base64)
        elif "image_data" in request.form:
            image_data = request.form["image_data"]
            # Tách phần header "data:image/jpeg;base64," ra khỏi chuỗi data
            if "," in image_data:
                image_data = image_data.split(",")[1]

            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            prediction, probs = predict_image(img)

    return render_template(
        "index.html",
        prediction=prediction,
        probs=probs,
        image_path=image_path
    )

# ======================
# Run server
# ======================
if __name__ == "__main__":
    app.run(debug=True)