# 🫁 PneumoVision AI  
## ResNet18-Based Pneumonia Detection System

This project implements a deep learning-based pneumonia detection system using **ResNet18** and **Flask**. It classifies chest X-ray images into:

- **Normal**
- **Bacterial Pneumonia**
- **Viral Pneumonia**

The system integrates model inference with a web interface and generates automated PDF diagnostic reports, simulating real-world AI-assisted medical screening systems.

---

## 🚀 Features

- ResNet18-based multi-class image classification  
- Transfer learning using pretrained ImageNet weights  
- Real-time prediction via Flask web interface  
- Confidence score display  
- Automated PDF report generation  
- End-to-end deep learning pipeline  

---

## 🧠 Tech Stack

- Python  
- PyTorch  
- Torchvision  
- Flask  
- OpenCV  
- FPDF  

---

## ⚙️ How It Works

- Chest X-ray images are preprocessed and resized to **224×224**  
- A pretrained ResNet18 model is fine-tuned for 3-class classification  
- The trained model performs real-time inference  
- Predictions and confidence scores are displayed  
- A downloadable PDF diagnostic report is generated  

---

## ▶ Commands To Run

pip install -r requirements.txt
python app.py

Open in browser:
http://127.0.0.1:5000
