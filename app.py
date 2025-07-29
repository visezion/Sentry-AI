#!/usr/bin/env python3
"""
SENTRY-AI Network Anomaly Detection System with Multi-Modal (Tabular + Visual)
Feature Visualization, Grad-CAM Explanations, and a Visualization Dashboard Endpoint.

This file integrates:
1. A Variational Autoencoder (VAE) for anomaly detection using tabular features.
2. A Reinforcement Learning (RL) agent (using stable_baselines3 DQN) for automated response.
3. Computer vision modules to convert time-series network traffic features into images,
   train a CNN model (using a robust training routine via ImageFolder), and generate Grad-CAM visualizations.
4. Multi-modal fusion to combine decisions from the VAE branch and the CNN branch.
5. A Flask web server with endpoints for dashboard, predictions, performance, and anomaly visualizations.
    
Author: Victor Ayodeji Oluwasusi
Date: 2025-03-28
"""

# =============================================================================
# IMPORTS
# =============================================================================
import glob, os, time, logging, signal, threading, json, sys, base64
from io import BytesIO

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt, seaborn as sns

from flask import Flask, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from gym import Env
from gym.spaces import Discrete, Box

import bcrypt, joblib

# Scapy for packet sniffing and sending
from scapy.all import sniff, IP, TCP, send

# Optional: Hyperparameter tuning library (Optuna)
try:
    import optuna
except ImportError:
    optuna = None
    print("Optuna is not installed; hyperparameter tuning will be disabled.")

# =============================================================================
# CONFIGURATION & GLOBAL CONSTANTS
# =============================================================================
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "supersecurekey")
DATASET_PATH = os.environ.get("DATASET_PATH", "dataset")
ANOMALY_MODEL_PATH = os.environ.get("ANOMALY_MODEL_PATH", "models/anomaly_detector_best.pth")
RL_MODEL_PATH = os.environ.get("RL_MODEL_PATH", "models/rl_model_best.zip")
PERFORMANCE_METRICS_PATH = os.environ.get("PERFORMANCE_METRICS_PATH", "models/performance_metrics.json")
NETWORK_INTERFACE = os.environ.get("NETWORK_INTERFACE", "Realtek PCIe GbE Family Controller")
TRAINING_SCALER_PATH = os.environ.get("TRAINING_SCALER_PATH", "models/training_scaler.pkl")
# Path for CNN dataset (expects subfolders 'benign' and 'anomaly')
CNN_DATASET_PATH = os.environ.get("CNN_DATASET_PATH", "cnn_dataset")
CNN_MODEL_PATH = os.environ.get("CNN_MODEL_PATH", "models/cnn_model_best.pth")

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_RL_TOTAL_TIMESTEPS = 500000
DEFAULT_RL_CALLBACK_INTERVAL = 500
ONLINE_UPDATE_TIMESTEPS = 500
ONLINE_UPDATE_INTERVAL = 60
LEARNING_RATE = 0.001
RL_TOTAL_TIMESTEPS = DEFAULT_RL_TOTAL_TIMESTEPS
RL_CALLBACK_INTERVAL = DEFAULT_RL_CALLBACK_INTERVAL

# VAE anomaly detection threshold
VAE_THRESHOLD = 0.01

# Fusion weights for multi-modal prediction
W_VAE = 0.5
W_CNN = 0.5

# Fixed image size for CNN and GAF conversion
IMAGE_SIZE = 28

# Dynamic threshold percentile for test endpoint
THRESHOLD_PERCENTILE = 90

# =============================================================================
# LOGGING SETUP
# =============================================================================
logger = logging.getLogger("SENTRY-AI")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
from logging.handlers import RotatingFileHandler
file_handler = RotatingFileHandler("sentry_ai.log", maxBytes=5*1024*1024, backupCount=2)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# =============================================================================
# FLASK APP & SOCKETIO SETUP WITH RATE LIMITER
# =============================================================================
app = Flask(__name__, template_folder="templates")
app.secret_key = FLASK_SECRET_KEY
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
limiter = Limiter(key_func=get_remote_address, app=app, storage_uri="memory://")

# Global shutdown event for graceful termination
shutdown_event = threading.Event()

# =============================================================================
# PATHS FOR SAVED DATA & MODELS
# =============================================================================
CLEANED_DATA_PATH = "cleaned_data/cleaned_data_v2.pkl"

# =============================================================================
# PERFORMANCE MONITORING CLASS
# =============================================================================
class AIMonitor:
    """Tracks performance metrics for anomaly loss (VAE) and RL rewards."""
    def __init__(self):
        self.anomaly_loss_history = []  # List of (epoch, loss)
        self.rl_reward_history = []     # List of (timestep, reward)
    def log_anomaly_loss(self, epoch, loss):
        self.anomaly_loss_history.append((epoch, loss))
    def log_rl_reward(self, timestep, reward):
        self.rl_reward_history.append((timestep, reward))
    def get_best_anomaly_loss(self):
        return min(self.anomaly_loss_history, key=lambda x: x[1]) if self.anomaly_loss_history else None
    def get_best_rl_reward(self):
        return max(self.rl_reward_history, key=lambda x: x[1]) if self.rl_reward_history else None
    def compute_growth_metrics(self):
        growth = {}
        if self.anomaly_loss_history:
            start_loss = self.anomaly_loss_history[0][1]
            current_loss = self.anomaly_loss_history[-1][1]
            improvement = ((start_loss - current_loss) / start_loss * 100) if start_loss != 0 else 0
            growth['anomaly_loss_improvement_pct'] = improvement
        if self.rl_reward_history:
            start_reward = self.rl_reward_history[0][1]
            current_reward = self.rl_reward_history[-1][1]
            improvement = ((current_reward - start_reward) / abs(start_reward) * 100) if start_reward != 0 else 0
            growth['rl_reward_improvement_pct'] = improvement
        return growth

monitor = AIMonitor()

def save_performance_metrics():
    """Saves performance metrics to a JSON file."""
    with open(PERFORMANCE_METRICS_PATH, "w") as f:
        json.dump({
            "anomaly_loss_history": monitor.anomaly_loss_history,
            "rl_reward_history": monitor.rl_reward_history
        }, f)
    logger.info("Performance metrics saved.")

def load_performance_metrics():
    """Loads performance metrics from a JSON file if available."""
    if os.path.exists(PERFORMANCE_METRICS_PATH):
        with open(PERFORMANCE_METRICS_PATH, "r") as f:
            metrics = json.load(f)
            monitor.anomaly_loss_history = metrics.get("anomaly_loss_history", [])
            monitor.rl_reward_history = metrics.get("rl_reward_history", [])
        logger.info("Performance metrics loaded.")

# =============================================================================
# DATA LOADING & PREPROCESSING FUNCTIONS
# =============================================================================
def clean_data(df, batch_size=100000):
    """
    Cleans the DataFrame by dropping unused columns, imputing missing values with KNN,
    and scaling numerical features.
    """
    logger.info("Cleaning dataset...")
    df.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'], errors='ignore', inplace=True)
    df.columns = df.columns.str.strip().str.lower()
    if 'label' not in df.columns:
        logger.error(f"Error: 'label' column not found! Available columns: {df.columns.tolist()}")
        raise KeyError("The dataset does not contain a 'label' column.")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = KNNImputer(n_neighbors=5)
    num_batches = len(df) // batch_size + 1
    df_processed = []
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch = df.iloc[start:end, :-1]
        batch_imputed = imputer.fit_transform(batch) if batch.isnull().sum().sum() > 0 else batch.to_numpy()
        df_processed.append(batch_imputed)
    df.iloc[:, :-1] = np.vstack(df_processed)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df.iloc[:, :-1])
    logger.info("Dataset cleaning completed.")
    return X_scaled, df['label']

def load_and_clean_data():
    """
    Loads the dataset from CSV files (if not cached), cleans it, and caches the result.
    """
    if os.path.exists(CLEANED_DATA_PATH):
        logger.info("Found saved cleaned dataset. Loading...")
        try:
            X, y = joblib.load(CLEANED_DATA_PATH)
            logger.info(f"Cleaned dataset loaded with features shape: {X.shape} and labels length: {len(y)}")
        except Exception as e:
            logger.error(f"Error loading cleaned dataset: {e}")
            raise
    else:
        logger.info("Cleaned dataset not found. Loading raw CSV datasets...")
        csv_files = glob.glob(os.path.join(DATASET_PATH, "*.csv"))
        if not csv_files:
            logger.error("No CSV files found in the dataset folder!")
            raise FileNotFoundError("No CSV files found in dataset folder.")
        df_list = []
        good = {"benign"}
        for file in csv_files:
            try:
                logger.info(f"Loading dataset: {file}")
                df_temp = pd.read_csv(file, low_memory=False)
                logger.info(f"Loaded {file} with shape: {df_temp.shape}")
                df_temp.columns = df_temp.columns.str.strip()
                if "label" in df_temp.columns:
                    df_temp["label"] = df_temp["label"].apply(lambda x: 0 if str(x).strip().lower() in good else 1)
                elif "Label" in df_temp.columns:
                    df_temp["Label"] = df_temp["Label"].apply(lambda x: 0 if str(x).strip().lower() in good else 1)
                    df_temp.rename(columns={"Label": "label"}, inplace=True)
                else:
                    logger.error("No label column found in file: " + file)
                    continue
                df_list.append(df_temp)
            except Exception as e:
                logger.error(f"Failed to load {file}: {e}")
        if not df_list:
            raise ValueError("No valid CSV files loaded!")
        df = pd.concat(df_list, ignore_index=True)
        logger.info(f"Concatenated dataset shape: {df.shape}")
        X, y = clean_data(df)
        joblib.dump((X, y), CLEANED_DATA_PATH)
        logger.info("Cleaned dataset saved for future use.")
    return X, y

# Load dataset and split into training and testing sets
logger.info("Loading dataset...")
X, y = load_and_clean_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info(f"Data split. Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

# Fit and save scaler based on training data for live predictions
training_scaler = MinMaxScaler()
X_train = training_scaler.fit_transform(X_train)
os.makedirs(os.path.dirname(TRAINING_SCALER_PATH), exist_ok=True)
joblib.dump(training_scaler, TRAINING_SCALER_PATH)
logger.info("Training scaler fitted and saved.")

# Load performance metrics if available
logger.info("Loading performance metrics...")
load_performance_metrics()

# =============================================================================
# AGGREGATE LIVE FEATURES FUNCTION
# =============================================================================
def aggregate_live_features(live_data, target_dim):
    """
    Aggregates live packet features by computing statistics (mean, std, min, max, median, percentiles)
    and returns a fixed-dimension feature vector.
    """
    if live_data.size == 0:
        logger.warning("Empty live data in aggregate_live_features; returning zeros.")
        return np.zeros(target_dim)
    stats = []
    num_features = live_data.shape[1]
    for i in range(num_features):
        col_data = live_data[:, i]
        stats.extend([
            np.mean(col_data),
            np.std(col_data),
            np.min(col_data),
            np.max(col_data),
            np.median(col_data),
            np.percentile(col_data, 25),
            np.percentile(col_data, 75)
        ])
    global_stats = [np.mean(live_data), np.std(live_data), np.min(live_data), np.max(live_data)]
    stats.extend(global_stats)
    means = [np.mean(live_data[:, i]) for i in range(num_features)]
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            stats.append(means[i] - means[j])
    stats = np.array(stats)
    if stats.shape[0] >= target_dim:
        return stats[:target_dim]
    else:
        return np.concatenate([stats, np.zeros(target_dim - stats.shape[0])])

# =============================================================================
# VARIATIONAL AUTOENCODER (VAE) FOR ANOMALY DETECTION
# =============================================================================
class VariationalAutoencoder(nn.Module):
    """A simple feed-forward VAE for anomaly detection."""
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.fc_decode1 = nn.Linear(latent_dim, 128)
        self.fc_decode2 = nn.Linear(128, input_dim)
        self.relu = nn.ReLU()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.relu(self.fc_decode1(z))
        return self.fc_decode2(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

detector = VariationalAutoencoder(X_train.shape[1])
optimizer = optim.Adam(detector.parameters(), lr=LEARNING_RATE)
mse_loss_fn = nn.MSELoss()

def kl_divergence(mu, logvar):
    """Computes the KL divergence loss."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def train_anomaly_detector():
    """Trains the VAE and emits updates via SocketIO."""
    logger.info("Starting VAE training...")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    epochs = 100
    for epoch in tqdm(range(epochs), desc="Training VAE"):
        detector.train()
        optimizer.zero_grad()
        reconstruction, mu, logvar = detector(X_train_tensor)
        mse_loss = mse_loss_fn(reconstruction, X_train_tensor)
        kl_loss = kl_divergence(mu, logvar) / X_train_tensor.shape[0]
        loss = mse_loss + 0.001 * kl_loss
        loss.backward()
        optimizer.step()
        current_loss = loss.item()
        monitor.log_anomaly_loss(epoch + 1, current_loss)
        tqdm.write(f"Epoch {epoch+1}/100, Loss: {current_loss:.4f}")
        socketio.emit("anomaly_loss_update", {"epoch": epoch + 1, "loss": current_loss})
    logger.info("VAE training completed.")

if os.path.exists(ANOMALY_MODEL_PATH):
    logger.info("Loading saved VAE model...")
    detector.load_state_dict(torch.load(ANOMALY_MODEL_PATH))
else:
    logger.info("No saved VAE model found. Training...")
    train_anomaly_detector()
    torch.save(detector.state_dict(), ANOMALY_MODEL_PATH)
    logger.info(f"VAE model saved after training. Best loss: {monitor.get_best_anomaly_loss()[1]:.4f}")


# =============================================================================
# CUSTOM REINFORCEMENT LEARNING ENVIRONMENT (CYBER DEFENSE)
# =============================================================================
class CyberDefenseEnv(Env):
    """Custom Gym environment simulating a network defense scenario."""
    def __init__(self, live_probability=1.0, max_steps=100):
        super().__init__()
        self.num_features = X_train.shape[1]
        self.action_space = Discrete(3)  # 0: Allow, 1: Alert, 2: Block
        self.observation_space = Box(low=0, high=1, shape=(self.num_features,), dtype=np.float32)
        self.live_probability = live_probability
        self.mode = "csv"
        self.state = np.random.rand(self.num_features)
        self.step_count = 0
        self.max_steps = max_steps
    def reset(self):
        self.mode = "live" if np.random.rand() < self.live_probability else "csv"
        self.step_count = 0
        self.state = np.random.rand(self.num_features)
        logger.info(f"Episode reset. Mode: {self.mode}")
        return self.state
    def step(self, action):
        if self.mode == "live":
            reward = 10 if action == 0 else (0 if action == 1 else -20)
            state = np.random.rand(self.num_features)
        else:
            base_reward = 10 if action == 2 else 5 if action == 1 else -2
            if np.random.rand() < 0.2:
                base_reward = -abs(base_reward)
            reward = base_reward * np.random.uniform(0.8, 1.2)
            state = np.random.rand(self.num_features)
        self.step_count += 1
        done = self.step_count >= self.max_steps
        return state, reward, done, {}

env = CyberDefenseEnv(live_probability=1.0, max_steps=100)

# =============================================================================
# CUSTOM REPLAY BUFFER
# =============================================================================
class CustomReplayBuffer:
    """Simple replay buffer for online RL training."""
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size
    def add(self, experience):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

custom_replay_buffer = CustomReplayBuffer(max_size=10000)

# =============================================================================
# RL REWARD CALLBACK
# =============================================================================
class RLRewardCallback(BaseCallback):
    """Callback to log and emit RL reward updates."""
    def __init__(self, monitor, verbose=0):
        super().__init__(verbose)
        self.monitor = monitor
    def _on_step(self) -> bool:
        if self.n_calls % RL_CALLBACK_INTERVAL == 0 and self.model.ep_info_buffer:
            avg_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            self.monitor.log_rl_reward(self.n_calls, avg_reward)
            best_reward_entry = self.monitor.get_best_rl_reward()
            best_reward = best_reward_entry[1] if best_reward_entry else None
            logger.info(f"[RL Reward Update] Timestep {self.n_calls}: Avg reward = {avg_reward:.2f}, Best = {best_reward}")
            socketio.emit("rl_reward_update", {"timestep": self.n_calls, "reward": avg_reward})
        return True

reward_callback = RLRewardCallback(monitor)

# =============================================================================
# LOAD OR TRAIN RL MODEL
# =============================================================================
policy_kwargs = {"dueling": True}
if os.path.exists(RL_MODEL_PATH):
    logger.info("Loading saved RL model...")
    model = DQN.load(RL_MODEL_PATH, env=env)
    new_logger = configure(folder=None, format_strings=["stdout"])
    model._logger = new_logger
else:
    logger.info("No saved RL model found. Training new RL model...")
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=RL_TOTAL_TIMESTEPS, callback=reward_callback)
    model.save(RL_MODEL_PATH)
    logger.info("RL model trained and saved.")

# =============================================================================
# ONLINE TRAINING LOOP
# =============================================================================
def online_training_loop():
    """Continuously trains the RL model using experiences from the replay buffer."""
    update_count = 0
    while not shutdown_event.is_set():
        try:
            if len(custom_replay_buffer.buffer) > 100:
                for exp in custom_replay_buffer.buffer:
                    state, action, reward, next_state, done = exp
                    model.replay_buffer.add(state, next_state, action, reward, done, [{}])
                model.learn(total_timesteps=ONLINE_UPDATE_TIMESTEPS, reset_num_timesteps=False, callback=reward_callback)
                update_count += 1
                logger.info(f"Online Training Update #{update_count} with {len(custom_replay_buffer.buffer)} experiences.")
                custom_replay_buffer.buffer = []
        except Exception as e:
            logger.error(f"Error in online_training_loop: {e}")
        time.sleep(ONLINE_UPDATE_INTERVAL)

# =============================================================================
# PERIODIC STATUS UPDATE
# =============================================================================
def periodic_status():
    """Logs best anomaly loss and RL reward every 60 seconds."""
    while not shutdown_event.is_set():
        best_loss_entry = monitor.get_best_anomaly_loss()
        best_rl_entry = monitor.get_best_rl_reward()
        logger.info(f"[Status Update] Best VAE loss: {best_loss_entry[1] if best_loss_entry else 'N/A'}, "
                    f"Best RL reward: {best_rl_entry[1] if best_rl_entry else 'N/A'}")
        time.sleep(60)

socketio.start_background_task(target=periodic_status)

# =============================================================================
# PACKET SNIFFING & LIVE DATA PROCESSING.
# =============================================================================
def extract_features(packet):
    """Extracts numerical features from a network packet."""
    if IP in packet:
        return [
            packet[IP].len,
            packet[IP].ttl,
            packet[IP].proto,
            packet[TCP].sport if TCP in packet else 0,
            packet[TCP].dport if TCP in packet else 0
        ]
    return None

def extract_packet_details(packet):
    """Extracts IP-level details from a network packet."""
    if IP in packet:
        return {"src": packet[IP].src, "dst": packet[IP].dst, "proto": packet[IP].proto}
    return {}

def packet_sniffer(count=100, iface=NETWORK_INTERFACE):
    """
    Captures 'count' packets from the specified interface.
    Returns a numpy array of features and a list of packet details.
    """
    logger.info("Starting packet sniffing on %s", iface)
    live_data = []
    details = []
    def process_packet(pkt):
        feat = extract_features(pkt)
        if feat is not None:
            live_data.append(feat)
            details.append(extract_packet_details(pkt))
    sniff(prn=process_packet, count=count, store=0, iface=iface)
    live_data = [f for f in live_data if f is not None]
    logger.info("Sniffing complete. Captured %d packets.", len(live_data))
    if details:
        logger.info("Sample packet detail: %s", details[0])
    return np.array(live_data), details

def process_live_data(data, skip_scaling=False):
    """Scales live data using the training scaler."""
    if skip_scaling:
        return data
    if os.path.exists(TRAINING_SCALER_PATH):
        scaler = joblib.load(TRAINING_SCALER_PATH)
    else:
        scaler = training_scaler
        joblib.dump(scaler, TRAINING_SCALER_PATH)
    return scaler.transform(data)

# =============================================================================
# COMPUTER VISION & MULTI-MODAL DETECTION MODULES
# =============================================================================
def convert_timeseries_to_gaf(time_series):
    """
    Converts a 1D time-series to a Gramian Angular Field (GAF) image.
    Normalizes the series to [-1, 1], converts to polar coordinates, and computes cosine summation.
    """
    scaler_local = MinMaxScaler(feature_range=(-1, 1))
    scaled_ts = scaler_local.fit_transform(time_series.reshape(-1, 1)).flatten()
    phi = np.arccos(scaled_ts)
    gaf = np.cos(phi[:, None] + phi[None, :])
    return gaf

class CNNAnomalyDetector(nn.Module):
    """
    A simple CNN for image-based anomaly detection.
    Assumes input images are single-channel of size IMAGE_SIZE x IMAGE_SIZE.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        flat_size = 32 * ((IMAGE_SIZE // 2) // 2) * ((IMAGE_SIZE // 2) // 2)
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Define global CNN model
cnn_model = CNNAnomalyDetector(num_classes=2)

def train_cnn_model_robust(epochs=20, batch_size=32, learning_rate=0.001):
    """
    Trains the CNN model using a robust labeled dataset.
    Expects the dataset to be structured in CNN_DATASET_PATH with subfolders 'benign' and 'anomaly'.
    """
    global cnn_model
    logger.info("Loading CNN dataset from %s", CNN_DATASET_PATH)
    if not os.path.isdir(CNN_DATASET_PATH):
        logger.error("CNN dataset folder not found. Create subfolders 'benign' and 'anomaly'.")
        return
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    val_transforms = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
    full_dataset = datasets.ImageFolder(root=CNN_DATASET_PATH, transform=train_transforms)
    if len(full_dataset) == 0:
        raise FileNotFoundError(f"No valid images found in {CNN_DATASET_PATH}.")
    dataset_size = len(full_dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transforms
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=learning_rate)
    best_val_acc = 0
    best_model_state = None
    cnn_model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer_cnn.zero_grad()
            outputs = cnn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_cnn.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / train_size
        cnn_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = cnn_model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_acc = correct / total
        logger.info(f"CNN Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = cnn_model.state_dict()
        cnn_model.train()
    if best_model_state is not None:
        torch.save(best_model_state, CNN_MODEL_PATH)
        logger.info("CNN model trained and saved with best Val Acc: {:.4f}".format(best_val_acc))
    else:
        logger.error("CNN training failed to produce a valid model.")

def generate_gradcam(model, input_tensor, target_class):
    """
    Generates a Grad-CAM heatmap for a given image tensor and target class.
    Hooks into the last convolutional layer to extract activations and gradients.
    """
    model.eval()
    activations = {}
    gradients = {}
    def forward_hook(module, input, output):
        activations['value'] = output.detach()
    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()
    hook_handle = model.features[-3].register_forward_hook(forward_hook)
    hook_handle_back = model.features[-3].register_backward_hook(backward_hook)
    output = model(input_tensor.unsqueeze(0))
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    model.zero_grad()
    output.backward(gradient=one_hot, retain_graph=True)
    pooled_gradients = torch.mean(gradients['value'], dim=[0, 2, 3])
    activation = activations['value'][0]
    for i in range(activation.shape[0]):
        activation[i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activation, dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    hook_handle.remove()
    hook_handle_back.remove()
    return heatmap

def multi_modal_predict(agg_features, image):
    """
    Fuses predictions from the VAE branch (tabular features) and the CNN branch (visual features).
    Computes a normalized VAE ratio and obtains the CNN anomaly probability via softmax.
    A weighted fusion score is computed; if it exceeds 1, the sample is classified as anomaly.
    
    Returns:
      final_decision: "anomaly" or "benign"
      vae_mse: Reconstruction error from VAE
      cnn_prob: CNN anomaly probability (class 1)
      fusion_score: Weighted fusion score
    """
    input_tensor = torch.tensor(agg_features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        rec, _, _ = detector(input_tensor)
        vae_mse = torch.mean((rec - input_tensor)**2).item()
    vae_ratio = vae_mse / VAE_THRESHOLD
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        cnn_output = cnn_model(image_tensor)
        cnn_prob = torch.softmax(cnn_output, dim=1)[0, 1].item()
    fusion_score = W_VAE * vae_ratio + W_CNN * cnn_prob
    final_decision = "anomaly" if fusion_score > 1 else "benign"
    return final_decision, vae_mse, cnn_prob, fusion_score

# =============================================================================
# LIVE DATA STREAM WITH MULTI-MODAL FUSION & RL ACTION
# =============================================================================
def live_data_stream():
    """
    Continuously captures live packet data, aggregates features, creates a visual representation,
    fuses predictions from VAE and CNN branches, and uses the RL model to decide an action.
    Emits live updates via SocketIO.
    """
    while not shutdown_event.is_set():
        try:
            live_data, packet_details = packet_sniffer(count=100, iface=NETWORK_INTERFACE)
            if live_data.size == 0:
                logger.warning("No live data captured; skipping iteration.")
                time.sleep(5)
                continue

            agg_features = aggregate_live_features(live_data, target_dim=X_train.shape[1])
            if agg_features.size == 0:
                logger.error("Aggregated features empty; skipping iteration.")
                time.sleep(5)
                continue

            if os.path.exists(TRAINING_SCALER_PATH):
                scaler = joblib.load(TRAINING_SCALER_PATH)
            else:
                scaler = training_scaler
                joblib.dump(scaler, TRAINING_SCALER_PATH)
            scaled_features = scaler.transform(agg_features.reshape(1, -1))[0]

            # Create a GAF image from aggregated features
            gaf = convert_timeseries_to_gaf(agg_features)
            image = Image.fromarray((gaf * 255).astype(np.uint8))
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

            final_decision, vae_mse, cnn_prob, fusion_score = multi_modal_predict(scaled_features, image)
            is_anomaly = (final_decision == "anomaly")

            action, _ = model.predict(scaled_features)
            if is_anomaly:
                reward = 10 if action == 2 else -10
                action_str = "Block" if action == 2 else ("Alert" if action == 1 else "Allow")
            else:
                reward = 10 if action == 0 else -10
                action_str = "Allow" if action == 0 else ("Alert" if action == 1 else "Block")

            current_state = scaled_features
            next_state, _, done, _ = env.step(action)
            custom_replay_buffer.add((current_state, action, reward, next_state, done))

            unique_ips = list({d.get("src") for d in packet_details if d.get("src")})
            protocol_counts = {}
            for d in packet_details:
                proto = d.get("proto")
                protocol_counts[proto] = protocol_counts.get(proto, 0) + 1

            update_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "packet_count": int(live_data.shape[0]),
                "aggregated_features": scaled_features.tolist(),
                "predicted_action": action_str,
                "is_anomaly": is_anomaly,
                "vae_mse": vae_mse,
                "cnn_probability": cnn_prob,
                "fusion_score": fusion_score,
                "interface_status": "active" if live_data.size > 0 else "inactive",
                "unique_ips": unique_ips,
                "protocol_distribution": protocol_counts
            }
            socketio.emit("live_update", update_data)
        except Exception as e:
            logger.error(f"Error in live_data_stream: {e}")
        time.sleep(5)

# =============================================================================
# OFFLINE RL TRAINING
# =============================================================================
def train_rl_offline(model, monitor, X, y, callback_interval=100, total_timesteps=10000):
    """Performs offline RL training using the cleaned dataset."""
    logger.info("Starting offline RL training...")
    y_numeric = pd.to_numeric(y, errors='coerce')
    offline_data = np.hstack([X, y_numeric.values.reshape(-1, 1)]).astype(np.float32)
    steps = 0
    episode = 0
    max_episodes = 100

    class OfflineRLCallback(BaseCallback):
        def __init__(self, monitor, interval=callback_interval, verbose=0):
            super().__init__(verbose)
            self.monitor = monitor
            self.interval = interval
        def _on_step(self) -> bool:
            if self.n_calls % self.interval == 0 and self.model.ep_info_buffer:
                avg_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                self.monitor.log_rl_reward(self.n_calls, avg_reward)
                logger.info(f"[Offline RL] Timestep {self.n_calls}, Avg reward: {avg_reward:.2f}")
                socketio.emit("rl_reward_update", {"timestep": self.n_calls, "reward": avg_reward})
            return True

    offline_callback = OfflineRLCallback(monitor=monitor, interval=callback_interval)
    while episode < max_episodes and steps < total_timesteps:
        np.random.shuffle(offline_data)
        for row in offline_data:
            current_state = row[:-1]
            action, _ = model.predict(current_state)
            label = row[-1]
            reward = 10 if (label == 1 and action == 2) or (label == 0 and action == 0) else -10
            next_state = current_state
            done = False
            model.replay_buffer.add(current_state, next_state, action, reward, done, [{}])
            model.train(gradient_steps=1)
            steps += 1
            if steps >= total_timesteps:
                break
        logger.info(f"[Offline RL] Episode {episode+1} finished.")
        episode += 1
    model.save(RL_MODEL_PATH)
    logger.info("Offline RL training complete. Model saved.")

# =============================================================================
# PORT SCAN SIMULATION
# =============================================================================
def simulate_port_scan():
    """Simulates a port scan by sending SYN packets to a target IP."""
    logger.info("Port scan simulation started.")
    target_ip = "192.168.1.10"
    while not port_scan_stop_event.is_set():
        for port in range(20, 1024):
            if port_scan_stop_event.is_set():
                break
            packet = IP(dst=target_ip) / TCP(dport=port, flags="S")
            send(packet, verbose=False)
        time.sleep(1)
    logger.info("Port scan simulation stopped.")

def start_port_scan():
    """Starts the port scan simulation in a separate thread."""
    global port_scan_thread, port_scan_stop_event
    if port_scan_thread is None or not port_scan_thread.is_alive():
        port_scan_stop_event.clear()
        port_scan_thread = threading.Thread(target=simulate_port_scan)
        port_scan_thread.start()
        logger.info("Port scan simulation thread started.")

def stop_port_scan():
    """Stops the port scan simulation."""
    global port_scan_stop_event, port_scan_thread
    if port_scan_thread is not None and port_scan_thread.is_alive():
        port_scan_stop_event.set()
        port_scan_thread.join()
        logger.info("Port scan simulation thread stopped.")

# =============================================================================
# FLASK ENDPOINTS
# =============================================================================
@app.route('/')
@limiter.limit("10 per minute")
def dashboard():
    """Renders the dashboard page."""
    return render_template('dashboard.html')

@app.route('/anomalies')
@limiter.limit("10 per minute")
def get_anomalies():
    """Returns anomaly scores computed on the test set using the VAE."""
    logger.info("Generating anomaly report...")
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    anomaly_scores = torch.mean((detector(X_test_tensor)[0] - X_test_tensor)**2, axis=1).detach().numpy()
    logger.info("Anomaly report generated.")
    return jsonify({"anomalies": anomaly_scores.tolist()})

@app.route('/predict')
@limiter.limit("10 per minute")
def predict_action():
    """
    Captures live network data, processes it, and predicts an action (Allow/Alert/Block)
    using the RL model.
    """
    logger.info("Predicting action from live data...")
    live_data, _ = packet_sniffer(count=100, iface=NETWORK_INTERFACE)
    if live_data.size == 0:
        logger.warning("No live data; defaulting to Allow.")
        return jsonify({"Action": "Allow"})
    processed_live = process_live_data(live_data)
    agg_features = aggregate_live_features(processed_live, target_dim=X_train.shape[1])
    action, _ = model.predict(agg_features)
    action_str = "Block" if action == 2 else "Alert" if action == 1 else "Allow"
    logger.info("Action prediction completed.")
    return jsonify({"Action": action_str})

@app.route('/performance')
@limiter.limit("10 per minute")
def get_performance():
    """Returns performance metrics for the VAE and RL agent."""
    best_loss_entry = monitor.get_best_anomaly_loss()
    best_loss = best_loss_entry[1] if best_loss_entry else None
    best_rl_entry = monitor.get_best_rl_reward()
    best_reward = best_rl_entry[1] if best_rl_entry else None
    performance_data = {
        "anomaly_loss_history": monitor.anomaly_loss_history,
        "rl_reward_history": monitor.rl_reward_history,
        "best_loss": best_loss,
        "best_reward": best_reward
    }
    return jsonify(performance_data)

@app.route('/ai_growth')
@limiter.limit("10 per minute")
def ai_growth():
    """Returns computed growth metrics for anomaly loss and RL reward improvement."""
    growth = monitor.compute_growth_metrics()
    best_anom = monitor.get_best_anomaly_loss()
    best_rl = monitor.get_best_rl_reward()
    growth.update({
        "last_best_anomaly_loss": best_anom[1] if best_anom else 'N/A',
        "last_best_rl_reward": best_rl[1] if best_rl else 'N/A'
    })
    return jsonify(growth)

@app.route('/start_port_scan', methods=['POST'])
@limiter.limit("5 per minute")
def start_port_scan_endpoint():
    """Starts the port scan simulation."""
    start_port_scan()
    return jsonify({"message": "Port scan simulation started."})

@app.route('/stop_port_scan', methods=['POST'])
@limiter.limit("5 per minute")
def stop_port_scan_endpoint():
    """Stops the port scan simulation."""
    stop_port_scan()
    return jsonify({"message": "Port scan simulation stopped."})

@app.route('/test_model', methods=['GET'])
@limiter.limit("10 per minute")
def test_model():
    """
    Tests the VAE on random samples from the test set and returns prediction results,
    computed MSE, and a dynamic threshold based on THRESHOLD_PERCENTILE.
    """
    y_test_array = y_test.values if hasattr(y_test, "values") else np.array(y_test)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        reconstruction, _, _ = detector(X_test_tensor)
        mse_scores = torch.mean((reconstruction - X_test_tensor)**2, axis=1).numpy()
    threshold = float(np.percentile(mse_scores, THRESHOLD_PERCENTILE))
    results = {}
    anomaly_indices = np.where(y_test_array == 1)[0]
    benign_indices = np.where(y_test_array == 0)[0]
    if len(anomaly_indices) > 0:
        idx_anomaly = int(np.random.choice(anomaly_indices))
        sample_anomaly = X_test[idx_anomaly]
        sample_tensor = torch.tensor(sample_anomaly, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            rec_anomaly, _, _ = detector(sample_tensor)
            mse_value_anomaly = torch.mean((rec_anomaly - sample_tensor)**2).item()
        predicted_anomaly = mse_value_anomaly > threshold
        results['anomaly_sample'] = {
            "index": idx_anomaly,
            "true_label": int(y_test_array[idx_anomaly]),
            "mse_value": mse_value_anomaly,
            "threshold": threshold,
            "predicted_anomaly": predicted_anomaly
        }
    else:
        results['anomaly_sample'] = "No anomaly samples found in test set"
    if len(benign_indices) > 0:
        idx_benign = int(np.random.choice(benign_indices))
        sample_benign = X_test[idx_benign]
        sample_tensor = torch.tensor(sample_benign, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            rec_benign, _, _ = detector(sample_tensor)
            mse_value_benign = torch.mean((rec_benign - sample_tensor)**2).item()
        predicted_anomaly = mse_value_benign > threshold
        results['benign_sample'] = {
            "index": idx_benign,
            "true_label": int(y_test_array[idx_benign]),
            "mse_value": mse_value_benign,
            "threshold": threshold,
            "predicted_anomaly": predicted_anomaly
        }
    else:
        results['benign_sample'] = "No benign samples found in test set"
    return jsonify(results)

@app.route('/live_predictions', methods=['GET'])
@limiter.limit("10 per minute")
def live_predictions():
    """
    Captures live packets and returns per-packet predictions including:
      - Raw features (as a list)
      - Aggregated features (scaled, as a list)
      - MSE anomaly score
      - Predicted RL action
    """
    live_data, details = packet_sniffer(count=50, iface=NETWORK_INTERFACE)
    predictions = []
    if live_data.size == 0:
        return jsonify({"predictions": "No packets captured"})
    for i, packet in enumerate(live_data):
        agg_feat = aggregate_live_features(np.array(packet).reshape(1, -1), target_dim=X_train.shape[1])
        scaled_feat = training_scaler.transform(agg_feat.reshape(1, -1))[0]
        input_tensor = torch.tensor(scaled_feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            rec, _, _ = detector(input_tensor)
            mse_val = torch.mean((rec - input_tensor)**2).item()
        anomaly_flag = mse_val > VAE_THRESHOLD
        action, _ = model.predict(scaled_feat)
        action_str = "Block" if action == 2 else "Alert" if action == 1 else "Allow"
        predictions.append({
            "packet_index": i,
            "raw_features": list(packet),
            "aggregated_features": scaled_feat.tolist(),
            "mse": mse_val,
            "anomaly": anomaly_flag,
            "predicted_action": action_str
        })
    return jsonify({"predictions": predictions})

@app.route('/visualize_anomalies', methods=['GET'])
@limiter.limit("10 per minute")
def visualize_anomalies():
    """
    Reads recent generated images from disk, base64 encodes their content along with a generated Grad-CAM heatmap,
    and returns them so the dashboard can display the real images.
    """
    image_folder = "generated_images"
    if not os.path.exists(image_folder):
        return jsonify({"error": "No generated images available."})
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")), reverse=True)[:5]
    visualizations = []
    for img_path in image_paths:
        try:
            with open(img_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            image = Image.open(img_path).convert("L")
            transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor()
            ])
            input_tensor = transform(image)
            heatmap = generate_gradcam(cnn_model, input_tensor, target_class=1)
            heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize((IMAGE_SIZE, IMAGE_SIZE))
            buffered = BytesIO()
            heatmap_img.save(buffered, format="PNG")
            heatmap_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            visualizations.append({
                "image": encoded_image,
                "heatmap": heatmap_base64,
                "predicted_action": "Block",  # Placeholder; update if needed
                "image_path": img_path.replace("\\", "/")
            })
        except Exception as ex:
            logger.error(f"Error processing image {img_path}: {ex}")
    return jsonify({"visualizations": visualizations})

# =============================================================================
# IMAGE DATASET GENERATION (BACKGROUND TASK)
# =============================================================================
def generate_image_dataset():
    """
    Continuously generates an image from aggregated live packet features,
    computes the anomaly score using the VAE, and saves the image with a filename
    containing the timestamp and predicted label.
    """
    image_folder = "generated_images"
    os.makedirs(image_folder, exist_ok=True)
    while not shutdown_event.is_set():
        try:
            live_data, _ = packet_sniffer(count=50, iface=NETWORK_INTERFACE)
            if live_data.size == 0:
                time.sleep(5)
                continue
            agg_features = aggregate_live_features(live_data, target_dim=X_train.shape[1])
            gaf = convert_timeseries_to_gaf(agg_features)
            img = Image.fromarray((gaf * 255).astype(np.uint8))
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            row_scaled = training_scaler.transform(agg_features.reshape(1, -1)).flatten()
            input_tensor = torch.tensor(row_scaled, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                rec, _, _ = detector(input_tensor)
                mse = torch.mean((rec - input_tensor)**2).item()
            label = "anomaly" if mse > VAE_THRESHOLD else "benign"
            filename = os.path.join(image_folder, f"{time.strftime('%Y%m%d_%H%M%S')}_{label}.png")
            img.save(filename)
            time.sleep(30)
        except Exception as e:
            logger.error(f"Error in image dataset generation: {e}")
            time.sleep(5)

# =============================================================================
# GRACEFUL SHUTDOWN HANDLER
# =============================================================================
def graceful_exit(signum, frame):
    """
    Signals shutdown, saves models and performance metrics, and exits gracefully.
    """
    logger.info("Gracefully exiting. Saving models and metrics...")
    shutdown_event.set()
    best_loss_entry = monitor.get_best_anomaly_loss()
    if best_loss_entry:
        torch.save(detector.state_dict(), ANOMALY_MODEL_PATH)
        logger.info(f"VAE saved with best loss: {best_loss_entry[1]:.4f}")
    else:
        logger.info("No VAE performance recorded; model not saved.")
    best_rl_entry = monitor.get_best_rl_reward()
    if best_rl_entry:
        model.save(RL_MODEL_PATH)
        logger.info(f"RL model saved with best reward: {best_rl_entry[1]:.2f}")
    else:
        logger.info("No RL reward recorded; model not saved.")
    save_performance_metrics()
    exit(0)

signal.signal(signal.SIGINT, graceful_exit)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    # Global variables for port scan simulation
    port_scan_thread = None
    from threading import Event
    port_scan_stop_event = Event()

    # --- CNN MODEL LOADING/TRAINING ---
    # Wrap CNN model loading/training in the main guard to avoid spawning processes on import.
    if os.path.exists(CNN_MODEL_PATH):
        logger.info("Loading saved CNN model...")
        cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH))
    else:
        logger.info("No saved CNN model found. Training robust CNN model...")
        train_cnn_model_robust(epochs=20)
        cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH))

    # --- OFFLINE RL TRAINING (Bootstrap) ---
    train_rl_offline(model, monitor, X, y, callback_interval=100, total_timesteps=5000)

    # --- START BACKGROUND TASKS ---
    socketio.start_background_task(target=live_data_stream)
    socketio.start_background_task(target=online_training_loop)
    socketio.start_background_task(target=generate_image_dataset)

    logger.info("Starting SENTRY-AI Web Dashboard with advanced RL training...")
    socketio.run(app, host="0.0.0.0", port=5001, debug=False, use_reloader=False)
