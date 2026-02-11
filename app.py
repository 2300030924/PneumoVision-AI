from flask import Flask, render_template, request, send_file, redirect, url_for
import torch
from torchvision import transforms, models
from PIL import Image
import os
import datetime
from fpdf import FPDF
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

HISTORY_FILE = "history.json"

# -----------------------------
# Load & Save History
# -----------------------------
def load_history():
    return json.load(open(HISTORY_FILE)) if os.path.exists(HISTORY_FILE) else []

def save_history():
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

history = load_history()

# -----------------------------
# Load Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 3)  # 3 classes: Normal, Bacterial, Viral
model.load_state_dict(torch.load("pneumonia_model.pth", map_location=device))
model.eval()

classes = ['Normal', 'Bacterial Pneumonia', 'Viral Pneumonia']

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return classes[predicted.item()], confidence.item() * 100, probs.squeeze().tolist()

def infection_stage(prediction, confidence):
    if prediction != 'Normal':
        if confidence >= 75:
            return "Severe Infection"
        elif confidence >= 50:
            return "Moderate Infection"
        else:
            return "Mild Infection"
    return "Healthy"

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction, confidence, filename, probs, message, stage = "", 0, "", [], "", ""
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            message = "No file uploaded"
        else:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            try:
                prediction, confidence, probs = predict_image(file_path)
                stage = infection_stage(prediction, confidence)

                # Save record
                record = {
                    "filename": filename,
                    "prediction": prediction,
                    "confidence": round(confidence, 2),
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "probs": probs,
                    "stage": stage
                }
                history.append(record)
                save_history()
            except Exception as e:
                print("Error:", e)
                message = "Error processing image"

    return render_template(
        "index.html",
        uploaded_image=filename if prediction else "",
        result=prediction,
        confidence=confidence,
        stage=stage,
        message=message,
        history=list(reversed(history))
    )

@app.route("/history")
def view_history():
    return render_template("history.html", history=list(reversed(history)))

@app.route("/download_pdf/<int:record_index>")
def download_pdf(record_index):
    if record_index >= len(history):
        return "Record not found", 404

    record = history[record_index]
    pdf = FPDF('P', 'mm', 'A4')
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 12, "Chest X-Ray Diagnosis Report", ln=True, align="C")
    pdf.ln(5)

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], record['filename'])
    if os.path.exists(img_path):
        pdf.image(img_path, x=55, w=90)

    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.cell(0, 8, f"Prediction: {record['prediction']}", ln=True)
    pdf.cell(0, 8, f"Confidence: {record['confidence']}%", ln=True)
    pdf.cell(0, 8, f"Stage: {record['stage']}", ln=True)
    pdf.cell(0, 8, f"Timestamp: {record['timestamp']}", ln=True)
    pdf.ln(5)

    if record['prediction'] == 'Bacterial Pneumonia':
        measures = "- Start antibiotics as prescribed\n- Maintain hydration\n- Seek immediate medical attention"
    elif record['prediction'] == 'Viral Pneumonia':
        measures = "- Rest & hydration\n- Use antivirals (if prescribed)\n- Monitor oxygen levels"
    else:
        measures = "- Lungs appear healthy\n- Continue normal monitoring"

    pdf.multi_cell(0, 7, f"Recommended Actions:\n{measures}")
    pdf_file = f"{record['filename'].split('.')[0]}_report.pdf"
    pdf.output(pdf_file)

    return send_file(pdf_file, as_attachment=True)

@app.route("/clear_history", methods=["POST"])
def clear_history():
    global history
    history = []
    save_history()
    return redirect(url_for('view_history'))

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
