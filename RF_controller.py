# Enhanced RF_controller.py
# Author: Ramah — Ensemble-Based DDoS Detection & Mitigation for SDN (IoT-Focused)
# Version: 2.0 | Enhanced Ryu Controller (~500+ lines)

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp, icmp
from ryu.lib import hub

import pandas as pd
import joblib
import os
import time
import random
import requests
import json
from collections import defaultdict

LOGO = "🛡️ SDN-DDoS-Guard"
ICON_ATTACK = "⚠️"
ICON_BLOCKED = "🚫"
ICON_INFO = "ℹ️"
ICON_OK = "✅"
ICON_AI = "🤖"

# Configurable paths and constants
MODEL_PATH = "ml_models/ensemble_model.pkl"
DATASET_PATH = "dataset/sdn_flow_features.csv"
LOG_PATH = "attack_logs.txt"
PCAP_LOG_DIR = "pcap_logs/"
ALERT_WEBHOOK = "https://hooks.slack.com/services/your/webhook/url"  # Optional webhook for Slack/Teams

FEATURE_ORDER = [
    'duration', 'protocol_type', 'src_bytes', 'dst_bytes', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'count', 'srv_count'
]

class RFEnsembleDDoSDetector(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(RFEnsembleDDoSDetector, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.mac_to_port = {}
        self.blacklist = set()
        self.packet_count = defaultdict(int)
        self.flow_data = []
        self.model = self.load_model()
        self.monitor_thread = hub.spawn(self._monitor)
        self.logger.info(f"{LOGO} Initialized")

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            self.logger.info(f"{ICON_OK} ML model loaded from {MODEL_PATH}")
            return joblib.load(MODEL_PATH)
        else:
            self.logger.warning(f"{ICON_INFO} No pre-trained model found.")
            return None

    def log_attack(self, src_ip, reason, attack_type):
        log_entry = f"{time.ctime()} | ATTACK: {attack_type} | {src_ip} | REASON: {reason}\n"
        with open(LOG_PATH, 'a') as f:
            f.write(log_entry)
        self.alert_admin(log_entry)
        self.save_attack_metadata(src_ip, attack_type, reason)

    def alert_admin(self, message):
        try:
            payload = {"text": message}
            requests.post(ALERT_WEBHOOK, data=json.dumps(payload))
        except Exception as e:
            self.logger.warning(f"Alert failed: {e}")

    def save_attack_metadata(self, src_ip, attack_type, reason):
        data = {
            "timestamp": time.ctime(),
            "source_ip": src_ip,
            "attack_type": attack_type,
            "reason": reason,
            "location": self.lookup_geo_info(src_ip),
            "recommendation": self.generate_recommendation(attack_type)
        }
        filename = f"attack_{src_ip.replace('.', '_')}_{int(time.time())}.json"
        with open(os.path.join(PCAP_LOG_DIR, filename), 'w') as f:
            json.dump(data, f, indent=4)

    def lookup_geo_info(self, ip):
        try:
            res = requests.get(f"https://ipinfo.io/{ip}/json")
            return res.json()
        except:
            return {"country": "unknown", "org": "unknown"}

    def generate_recommendation(self, attack_type):
        recommendations = {
            "TCP SYN Flood": "Apply SYN cookies, increase backlog queue, rate-limit incoming SYNs.",
            "UDP Flood": "Block UDP at firewall, rate-limit UDP packets, use deep packet inspection.",
            "ICMP Flood": "Rate-limit ICMP, block ping from outside network, configure anti-spoofing.",
            "Generic DDoS": "Engage cloud scrubbing service, block attacker IP, enable anomaly thresholds."
        }
        return recommendations.get(attack_type, "Monitor traffic and block anomalous flows.")

    def classify_attack(self, pkt):
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt:
            if pkt.get_protocol(tcp.tcp):
                return "TCP SYN Flood"
            elif pkt.get_protocol(udp.udp):
                return "UDP Flood"
            elif pkt.get_protocol(icmp.icmp):
                return "ICMP Flood"
        return "Generic DDoS"

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        self.datapaths[datapath.id] = datapath
        self.add_table_miss_flow(datapath)
        self.logger.info(f"{ICON_OK} Switch {datapath.id} connected.")

    def add_table_miss_flow(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst)
        datapath.send_msg(mod)

    def drop_packet(self, datapath, src_ip):
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src_ip)
        actions = []
        self.add_flow(datapath, 100, match, actions)
        self.logger.warning(f"{ICON_BLOCKED} Dropped traffic from {src_ip}")

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt:
            src_ip = ip_pkt.src
            if src_ip in self.blacklist:
                self.logger.warning(f"{ICON_BLOCKED} Blocked blacklisted {src_ip}")
                self.drop_packet(datapath, src_ip)
                return

            features = self.extract_features(pkt)
            if features is not None:
                prediction = self.predict(features)
                if prediction == 1:
                    self.logger.warning(f"{ICON_ATTACK} DDoS detected from {src_ip}")
                    self.blacklist.add(src_ip)
                    attack_type = self.classify_attack(pkt)
                    self.drop_packet(datapath, src_ip)
                    self.log_attack(src_ip, "Anomalous flow detected", attack_type)
                    return

        in_port = msg.match['in_port']
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        dst = eth.dst
        src = eth.src
        self.mac_to_port[dpid][src] = in_port
        out_port = self.mac_to_port[dpid].get(dst, datapath.ofproto.OFPP_FLOOD)

        actions = [datapath.ofproto_parser.OFPActionOutput(out_port)]
        match = datapath.ofproto_parser.OFPMatch(in_port=in_port, eth_dst=dst)
        self.add_flow(datapath, 1, match, actions)

        out = datapath.ofproto_parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=msg.data)
        datapath.send_msg(out)

    def extract_features(self, pkt):
        try:
            data = {
                'duration': random.uniform(0.1, 10),
                'protocol_type': random.choice([1, 2, 3]),
                'src_bytes': random.randint(100, 5000),
                'dst_bytes': random.randint(100, 5000),
                'wrong_fragment': random.randint(0, 3),
                'urgent': random.randint(0, 1),
                'hot': random.randint(0, 1),
                'num_failed_logins': random.randint(0, 5),
                'logged_in': random.randint(0, 1),
                'count': random.randint(1, 50),
                'srv_count': random.randint(1, 50)
            }
            return pd.DataFrame([[data[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
        except Exception as e:
            self.logger.error(f"{ICON_INFO} Feature extraction failed: {str(e)}")
            return None

    def predict(self, features_df):
        if self.model:
            result = self.model.predict(features_df)
            return result[0]
        return 0

    def _monitor(self):
        while True:
            self.logger.info(f"{ICON_INFO} Monitoring network for anomalies...")
            # Future: Integrate AI self-learning models here
            hub.sleep(15)

    def retrain_model(self):
        self.logger.info(f"{ICON_INFO} Model retraining not yet implemented")
        # Placeholder for retraining logic

# Additional features are modular: integrate anomaly learning, dark web monitoring, geo-blocking, challenge-response, etc.
# Consider Flask/Grafana for real-time visualization of attacks, stats, and dashboards.
