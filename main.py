from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import numpy as np
import joblib
import json

app = FastAPI()

# Load model and assets
model = tf.keras.models.load_model("coderead_no_indent.h5", custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU})
with open("tokenizer.json") as f:
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.load(f))
label_encoder = joblib.load("label_encoder.pkl")

# Setup static and template folders
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Readability prediction function
def predict_readability(code):
    seq = tokenizer.texts_to_sequences([code])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, padding='post', maxlen=1000)
    pred = model.predict(padded)
    class_idx = np.argmax(pred, axis=1)[0]
    class_label = label_encoder.inverse_transform([class_idx])[0]
    confidence = float(np.max(pred))
    return class_label, confidence

# Time and space complexity estimation
def estimate_complexity(code):
    time_complexity = "O(1)"
    space_complexity = "O(1)"

    loop_count = code.count("for") + code.count("while")
    if loop_count >= 2:
        time_complexity = "O(n^2)"
    elif loop_count == 1:
        time_complexity = "O(n)"

    if "new" in code or "[" in code or "]" in code:
        space_complexity = "O(n)"

    return time_complexity, space_complexity

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/analyze")
async def analyze_file(request: Request, codefile: UploadFile = File(...)):
    if not codefile.filename.endswith(".java"):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": "❌ Only .java files are supported.",
            "confidence": "N/A",
            "file": codefile.filename,
            "time": "-",
            "space": "-",
            "code": "(file type not supported)"
        })

    content = await codefile.read()
    try:
        code = content.decode("utf-8")
    except UnicodeDecodeError:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": "❌ File encoding not supported.",
            "confidence": "N/A",
            "file": codefile.filename,
            "time": "-",
            "space": "-",
            "code": "(could not decode file)"
        })

    label, conf = predict_readability(code)
    time_complexity, space_complexity = estimate_complexity(code)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": label,
        "confidence": f"{conf:.2%}",
        "file": codefile.filename,
        "time": time_complexity,
        "space": space_complexity,
        "code": code
    })
