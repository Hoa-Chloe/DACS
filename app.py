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
    active_tab = "upload"  # Mặc định khi mới vào là tab upload
    camera_image = None    # Biến để lưu lại ảnh camera sau khi load trang

    if request.method == "POST":
        # 1. Trường hợp upload ảnh từ máy
        file = request.files.get("file")
        if file and file.filename != "":
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(save_path)
            
            img = Image.open(save_path).convert("RGB")
            prediction, probs = predict_image(img)
            
            # Cấp đường dẫn cho HTML hiển thị lại ảnh đã tải lên
            image_path = f"static/uploads/{file.filename}"
            active_tab = "upload"

        # 2. Trường hợp chụp ảnh từ camera
        elif "image_data" in request.form:
            image_data = request.form["image_data"]
            camera_image = image_data  # Gửi ngược lại mã Base64 cho HTML
            
            if "," in image_data:
                base64_str = image_data.split(",")[1]
            else:
                base64_str = image_data

            image_bytes = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            prediction, probs = predict_image(img)
            active_tab = "camera"  # Đánh dấu tab camera đang hoạt động

    return render_template(
        "index.html",
        prediction=prediction,
        probs=probs,
        image_path=image_path,
        camera_image=camera_image,
        active_tab=active_tab
    )

# ======================
# Run server
# ======================
if __name__ == "__main__":
    app.run(debug=True)