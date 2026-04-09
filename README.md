# 🛸 AI-Based Multi-Drone Wildlife Monitoring & Intrusion Detection System

## 📌 Overview
This project presents an AI-driven, software-based multi-drone monitoring system designed to address human–wildlife conflict and wildlife conservation challenges. The system uses aerial imagery (real or simulated) to detect, track, and analyze wildlife and human activities, enabling early detection of urban wildlife intrusion and potential poaching threats.

---

## 🎯 Problem Statement
Rapid urbanization and habitat fragmentation have increased human–wildlife conflict and poaching threats. Traditional monitoring methods are manual and limited, leading to delayed detection. This project proposes an intelligent aerial monitoring system using AI and simulated multi-drone coordination to support proactive conservation and urban safety.

---

## 🚀 Features

### 🐾 Wildlife Detection & Tracking
Detects and tracks animals using deep learning models.

### 👤 Human & Poacher Detection
Identifies suspicious human activities in restricted zones.

### 📊 Movement Analysis & Prediction
Analyzes animal movement patterns and predicts high-risk zones.

### 🛸 Multi-Drone Coordination (Simulated)
Allocates surveillance areas and avoids redundancy across drones.

### 📡 Urban Wildlife Intrusion Detection
Detects wildlife entering urban areas and triggers alerts.

### 📈 Interactive Dashboard
Visualizes detections, tracking, and risk alerts in real-time.

---

## 🧠 Methodology

### 1. Data Collection & Preprocessing
- Use UAV/wildlife datasets and simulated drone feeds  
- Perform cleaning, annotation, and normalization  

### 2. Object Detection
- YOLOv8 model for detecting animals and humans  

### 3. Tracking
- DeepSORT for tracking movement across frames  

### 4. Behavior Analysis
- Identify suspicious patterns (poaching indicators)  

### 5. Multi-Drone Coordination
- Simulated environment for zone allocation and coverage  

### 6. Predictive Analysis
- Identify high-risk zones using spatio-temporal patterns  

### 7. Visualization
- Dashboard for real-time monitoring and alerts  

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Deep Learning:** PyTorch / TensorFlow  
- **Computer Vision:** OpenCV, YOLOv8 (Ultralytics)  
- **Tracking:** DeepSORT  
- **Data Handling:** NumPy, Pandas  
- **Visualization:** Streamlit / Flask  
- **Development:** Jupyter Notebook / Google Colab  

---
```
## 📂 Project Structure
project/
│── data/ # Datasets (images/videos)
│── models/ # Trained models
│── src/
│ ├── detection.py # YOLO detection
│ ├── tracking.py # DeepSORT tracking
│ ├── coordination.py # Multi-drone logic
│ ├── prediction.py # Risk analysis
│── app.py # Dashboard application
│── requirements.txt
│── README.md

```
---

## 📊 Evaluation Metrics

- Accuracy  
- Precision & Recall  
- mAP (Mean Average Precision)  
- Tracking Accuracy  
- Confusion Matrix  

---

## 📦 Deliverables

- Wildlife detection and tracking system  
- Poacher detection module  
- Urban intrusion alert system  
- Multi-drone coordination (simulation)  
- Interactive dashboard  

---

## 🔍 Scope
This project focuses on software-based analysis using aerial data and simulated drones. It does not involve physical drone control and serves as a decision-support system for wildlife conservation and urban safety.

---

## 🌍 Applications

- Forest and wildlife conservation  
- Anti-poaching surveillance  
- Urban safety monitoring  
- Smart city integration  
- Disaster and environmental monitoring  

---

## 📚 Future Work

- Integration with real-time drone feeds  
- Thermal imaging for night detection  
- GPS-based wildlife tracking  
- Real-world multi-drone deployment  

---

## 👥 Team

- Venkatesh Paitwar  
- Sandarbh Singh
- Srushti Agrawal   
- Shravani Borde  

---

## 📌 Note
This project is developed as a **Third Year Project (Data Science / AI domain)** and focuses on real-world problem-solving using modern deep learning techniques.
