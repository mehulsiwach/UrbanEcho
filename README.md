# UrbanEcho 🌆  
**Real-time Noise Pollution Mapping using AI and Crowdsourced Audio Data**

UrbanEcho is an AI-powered web application that detects, classifies, and visualizes urban noise pollution in real time using user-submitted audio and GPS data. The platform leverages deep learning (YAMNet + Custom CNN) for sound classification and displays live noise intensity and types on an interactive map powered by Leaflet.js.  

---

## 🚀 Features
- **Real-time Audio Classification:** Detect various urban noise types (traffic, sirens, construction, crowd, nature, etc.) using YAMNet + a fine-tuned CNN.  
- **Crowdsourced Mapping:** Users upload short audio clips with GPS coordinates, contributing to a real-time community noise map.  
- **Live Synchronization:** Firebase Realtime Database syncs location-based noise data instantly across connected users.  
- **Dynamic Visualization:** Leaflet.js map updates with colored markers indicating sound intensity and category.  
- **Scalable Cloud Deployment:** Containerized with Docker and deployed on Google Cloud Run for auto-scaling, GPU-ready hosting.  

---

## 🧠 Tech Stack
- **Backend:** Python, Flask  
- **Machine Learning:** TensorFlow, YAMNet, Custom CNN  
- **Frontend:** HTML, CSS, JavaScript, Leaflet.js  
- **Database:** Firebase Realtime Database  
- **Cloud & Deployment:** Docker, Google Cloud Run  
- **Others:** NumPy, Pandas, Librosa, SoundFile  

---

## 📊 System Architecture
1. **Client (Frontend):** Captures audio + geolocation, sends to Flask API  
2. **Flask Server:** Preprocesses audio and runs inference using YAMNet + CNN model  
3. **Firebase Realtime Database:** Stores classification outputs (label, confidence, coordinates)  
4. **Frontend Map (Leaflet.js):** Fetches noise data from Firebase and updates dynamic markers in real time  

**Workflow Diagram (Simplified):**  
User Mic → Flask API → ML Model → Firebase → Leaflet Map  

---

## 🧩 Setup & Installation

### Prerequisites
- Python 3.9+  
- Docker (for containerized deployment)  
- Firebase project with Realtime Database enabled  

### Steps
1. **Clone the repository**
