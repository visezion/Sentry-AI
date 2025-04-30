# utils/network_capture.py

from scapy.all import sniff, IP, TCP
from utils.logger import log_progress

def extract_features(packet):
    if IP in packet:
        return [
            packet[IP].len,
            packet[IP].ttl,
            packet[IP].proto,
            packet[TCP].sport if TCP in packet else 0,
            packet[TCP].dport if TCP in packet else 0
        ]
    return None

def packet_sniffer():
    log_progress("Starting real-time packet sniffing...")
    live_data = []
    sniff(prn=lambda pkt: live_data.append(extract_features(pkt)), count=500, store=0)
    log_progress("Packet sniffing completed.")
    return live_data
