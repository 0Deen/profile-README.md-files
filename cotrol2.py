from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub

import switch
from datetime import datetime
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

class SimpleMonitor13(switch.SimpleSwitch13):
    feature_columns = [
        'tp_src', 'tp_dst', 'ip_proto', 'icmp_code', 'icmp_type',
        'flow_duration_sec', 'flow_duration_nsec',
        'idle_timeout', 'hard_timeout', 'flags',
        'packet_count', 'byte_count',
        'packet_count_per_second', 'packet_count_per_nsecond',
        'byte_count_per_second', 'byte_count_per_nsecond'
    ]

    attack_explanations = {
        "UDP_FLOOD": {
            "why": "Massive number of UDP packets sent to random ports.",
            "prevent": "Enable UDP rate limiting, use firewall rules.",
            "objective": "Prevent bandwidth exhaustion due to UDP floods."
        },
        "SYN_FLOOD": {
            "why": "Exploits TCP handshake with half-open connections.",
            "prevent": "Use SYN cookies, rate limit SYN packets.",
            "objective": "Mitigate TCP handshake abuse attacks."
        },
        "HTTP_FLOOD": {
            "why": "Floods web server with HTTP requests.",
            "prevent": "Deploy WAF, validate user-agents, CAPTCHA.",
            "objective": "Protect application layer services."
        },
        "DNS_AMPLIFICATION": {
            "why": "Small query triggers large DNS responses to victim.",
            "prevent": "Configure DNS servers to prevent recursion abuse.",
            "objective": "Eliminate misuse of open resolvers."
        },
        # Extend with all 40+ attack types you listed as needed.
    }

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        self.label_encoder = LabelEncoder()
        self.flow_model = None
        self.blacklist = set()
        self.attack_log_file = "attack_log.txt"
        self.model_reload_interval = 300
        hub.spawn(self._periodic_retrain)

        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        print("🚀 Training time:", (end - start))

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info(f"📡 Registered datapath: {datapath.id}")
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.warning(f"⚠️ Unregistered datapath: {datapath.id}")
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)
            self.flow_predict()

    def _request_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        timestamp = datetime.now().timestamp()
        with open("PredictFlowStatsfile.csv", "w") as file0:
            file0.write(','.join(['timestamp', 'datapath_id', 'flow_id', 'ip_src', 'tp_src', 'ip_dst', 'tp_dst',
                                  'ip_proto', 'icmp_code', 'icmp_type', 'flow_duration_sec', 'flow_duration_nsec',
                                  'idle_timeout', 'hard_timeout', 'flags', 'packet_count', 'byte_count',
                                  'packet_count_per_second', 'packet_count_per_nsecond',
                                  'byte_count_per_second', 'byte_count_per_nsecond']) + '\n')
            for stat in sorted([flow for flow in ev.msg.body if flow.priority == 1],
                               key=lambda f: (f.match['eth_type'], f.match['ipv4_src'], f.match['ipv4_dst'], f.match['ip_proto'])):
                ip_src = stat.match.get('ipv4_src', '0.0.0.0')
                ip_dst = stat.match.get('ipv4_dst', '0.0.0.0')
                ip_proto = stat.match.get('ip_proto', 0)
                icmp_code = stat.match.get('icmpv4_code', -1)
                icmp_type = stat.match.get('icmpv4_type', -1)
                tp_src = stat.match.get('tcp_src', stat.match.get('udp_src', 0))
                tp_dst = stat.match.get('tcp_dst', stat.match.get('udp_dst', 0))
                flow_id = f"{ip_src}-{tp_src}-{ip_dst}-{tp_dst}-{ip_proto}"

                def safe_div(x, y): return x / y if y else 0

                file0.write(f"{timestamp},{ev.msg.datapath.id},{flow_id},{ip_src},{tp_src},{ip_dst},{tp_dst},"
                            f"{ip_proto},{icmp_code},{icmp_type},{stat.duration_sec},{stat.duration_nsec},"
                            f"{stat.idle_timeout},{stat.hard_timeout},{stat.flags},{stat.packet_count},{stat.byte_count},"
                            f"{safe_div(stat.packet_count, stat.duration_sec)},"
                            f"{safe_div(stat.packet_count, stat.duration_nsec)},"
                            f"{safe_div(stat.byte_count, stat.duration_sec)},"
                            f"{safe_div(stat.byte_count, stat.duration_nsec)}\n")

    def flow_training(self):
        self.logger.info("🧠 Flow Training Started...")
        if not os.path.exists("FlowStatsfile.csv"):
            self.logger.error("❌ FlowStatsfile.csv not found!")
            return

        df = pd.read_csv("FlowStatsfile.csv")
        if df.empty or 'label' not in df.columns:
            self.logger.error("❌ FlowStatsfile.csv is empty or missing 'label'")
            return

        try:
            X = df[self.feature_columns].astype("float64")
            y = self.label_encoder.fit_transform(df["label"])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            clf = RandomForestClassifier(n_estimators=100, random_state=0)
            self.flow_model = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            self.logger.info("✅ Training Complete")
            self.logger.info("📊 Confusion Matrix:\n" + str(cm))
            self.logger.info(f"🎯 Accuracy: {acc*100:.2f}% | ❌ Error: {(1-acc)*100:.2f}%")

            self._export_results(y_test, y_pred)

        except Exception as e:
            self.logger.error(f"⚠️ Training Error: {e}")

    def flow_predict(self):
        try:
            if not os.path.exists("PredictFlowStatsfile.csv"):
                self.logger.warning("📁 PredictFlowStatsfile.csv not found.")
                return

            df = pd.read_csv("PredictFlowStatsfile.csv")
            if df.empty:
                self.logger.warning("🕳️ PredictFlowStatsfile.csv is empty.")
                return

            X = df[self.feature_columns].astype("float64")
            preds = self.flow_model.predict(X)
            labels = self.label_encoder.inverse_transform(preds)

            benign = sum(1 for label in labels if label == "BENIGN")
            attacks = len(labels) - benign

            if benign / len(labels) * 100 > 80:
                self.logger.info("🟢 Legitimate traffic detected.")
            else:
                self.logger.warning("🔴 DDoS attack suspected!")
                for idx, label in enumerate(labels):
                    if label != "BENIGN":
                        ip = df.iloc[idx]['ip_dst']
                        self._blacklist_host(ip)
                        self._log_ddos_attack(ip, label)
                        self._describe_attack(label)

            with open("PredictFlowStatsfile.csv", "w") as f:
                f.write(','.join(df.columns) + '\n')

        except Exception as e:
            self.logger.error(f"⚠️ Prediction Error: {e}")

    def _log_ddos_attack(self, ip_dst, attack_type):
        with open(self.attack_log_file, "a") as log:
            log.write(f"{datetime.now()} - DDoS target: {ip_dst} | Attack Type: {attack_type}\n")

    def _describe_attack(self, attack_type):
        info = self.attack_explanations.get(attack_type.upper())
        if info:
            self.logger.info(f"🧠 Attack Type: {attack_type}")
            self.logger.info(f"❓ Why: {info['why']}")
            self.logger.info(f"🛡️ Prevention: {info['prevent']}")
            self.logger.info(f"🎯 Objective: {info['objective']}")
        else:
            self.logger.info(f"ℹ️ No description available for {attack_type}")

    def _blacklist_host(self, ip):
        if ip not in self.blacklist:
            self.blacklist.add(ip)
            self.logger.info(f"⛔ Blacklisted {ip}")

    def _periodic_retrain(self):
        while True:
            hub.sleep(self.model_reload_interval)
            self.logger.info("♻️ Retraining ML model...")
            self.flow_training()

    def _export_results(self, y_true, y_pred):
        try:
            pd.DataFrame({'actual': y_true, 'predicted': y_pred}).to_csv("detection_report.csv", index=False)
            self.logger.info("📤 Detection report saved as detection_report.csv")
        except Exception as e:
            self.logger.error(f"📛 Export Error: {e}")

    def evaluate_baseline(self):
        self.logger.info("🧪 Evaluating basic packet rate thresholding...")
        try:
            df = pd.read_csv("FlowStatsfile.csv")
            if 'packet_count_per_second' not in df or 'label' not in df:
                self.logger.warning("⛔ Required columns missing.")
                return
            threshold = 1000
            y_true = df["label"]
            y_pred = ['DDoS' if p > threshold else 'BENIGN' for p in df["packet_count_per_second"]]
            acc = accuracy_score(y_true, y_pred)
            self.logger.info(f"🧮 Baseline Accuracy: {acc*100:.2f}%")
        except Exception as e:
            self.logger.error(f"⚠️ Baseline Eval Error: {e}")
