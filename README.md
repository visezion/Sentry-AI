# 🛡️ SENTRY-AI: Intelligent Multi-Modal Network Anomaly Detection System

**Author:** Victor Ayodeji Oluwasusi  
**Version:** 1.0.0  
**Date:** April 2025  

**SENTRY-AI** is an advanced, real-time **AI-powered Network Intrusion Detection System (NIDS)** that combines **deep learning**, **computer vision**, and **reinforcement learning** to detect, explain, and autonomously respond to cyber threats.

By fusing tabular network traffic data with visual pattern recognition (via GAF image transformations), SENTRY-AI delivers precise, explainable, and actionable anomaly detection — all within an interactive Flask dashboard.

---

## 🚀 Key Features

🔍 **Multi-Modal Threat Detection**  
- Tabular analysis using a **Variational Autoencoder (VAE)**  
- Visual detection via **CNN trained on Gramian Angular Field (GAF) images**

🧠 **Explainable AI (XAI)**  
- Integrated **Grad-CAM** heatmaps for transparent CNN predictions

🤖 **Autonomous Cyber Defense**  
- **Reinforcement Learning agent (DQN)** decides whether to **Allow, Alert, or Block** traffic

📡 **Live Network Monitoring**  
- Real-time **packet sniffing** and feature extraction using **Scapy**

🖥️ **Interactive Web Dashboard**  
- Built with **Flask + SocketIO**, providing real-time insights, anomaly scores, visualizations, and controls

🧪 **Attack Simulation**  
- Built-in **port scan simulator** for stress-testing the detection pipeline

📈 **Continuous Learning**  
- Supports both **offline training** and **online RL updates** to adapt over time

---

## ⚙️ Requirements

- Python 3.8+
- `pip install -r requirements.txt`
- (Optional) CUDA-enabled GPU for faster model training
- Datasets in labeled CSV format (e.g., **CICIDS-2017**, **NSL-KDD**, **UNSW-NB15**)

---

## 📦 Installation

```bash
git clone https://github.com/visezion/sentry-ai.git
cd sentry-ai
pip install -r requirements.txt
