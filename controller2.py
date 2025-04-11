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

    # Explicitly define which features to use
    feature_columns = [
        'tp_src', 'tp_dst', 'ip_proto', 'icmp_code', 'icmp_type',
        'flow_duration_sec', 'flow_duration_nsec',
        'idle_timeout', 'hard_timeout', 'flags',
        'packet_count', 'byte_count',
        'packet_count_per_second', 'packet_count_per_nsecond',
        'byte_count_per_second', 'byte_count_per_nsecond'
    ]

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        self.label_encoder = LabelEncoder()
        self.flow_model = None

        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        print("Training time:", (end - start))

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)
            self.flow_predict()

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        timestamp = datetime.now().timestamp()
        with open("PredictFlowStatsfile.csv", "w") as file0:
            file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
            body = ev.msg.body
            for stat in sorted([flow for flow in body if flow.priority == 1], key=lambda flow:
                (flow.match['eth_type'], flow.match['ipv4_src'], flow.match['ipv4_dst'], flow.match['ip_proto'])):

                ip_src = stat.match.get('ipv4_src', '0.0.0.0')
                ip_dst = stat.match.get('ipv4_dst', '0.0.0.0')
                ip_proto = stat.match.get('ip_proto', 0)
                icmp_code = stat.match.get('icmpv4_code', -1)
                icmp_type = stat.match.get('icmpv4_type', -1)
                tp_src = stat.match.get('tcp_src', stat.match.get('udp_src', 0))
                tp_dst = stat.match.get('tcp_dst', stat.match.get('udp_dst', 0))
                flow_id = f"{ip_src}-{tp_src}-{ip_dst}-{tp_dst}-{ip_proto}"

                try:
                    packet_count_per_second = stat.packet_count / stat.duration_sec
                except ZeroDivisionError:
                    packet_count_per_second = 0

                try:
                    packet_count_per_nsecond = stat.packet_count / stat.duration_nsec
                except ZeroDivisionError:
                    packet_count_per_nsecond = 0

                try:
                    byte_count_per_second = stat.byte_count / stat.duration_sec
                except ZeroDivisionError:
                    byte_count_per_second = 0

                try:
                    byte_count_per_nsecond = stat.byte_count / stat.duration_nsec
                except ZeroDivisionError:
                    byte_count_per_nsecond = 0

                file0.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n"
                            .format(timestamp, ev.msg.datapath.id, flow_id, ip_src, tp_src, ip_dst, tp_dst,
                                    ip_proto, icmp_code, icmp_type,
                                    stat.duration_sec, stat.duration_nsec,
                                    stat.idle_timeout, stat.hard_timeout,
                                    stat.flags, stat.packet_count, stat.byte_count,
                                    packet_count_per_second, packet_count_per_nsecond,
                                    byte_count_per_second, byte_count_per_nsecond))

    def flow_training(self):
        self.logger.info("Flow Training ...")

        if not os.path.exists('FlowStatsfile.csv'):
            self.logger.error("FlowStatsfile.csv not found!")
            return

        flow_dataset = pd.read_csv('FlowStatsfile.csv')
        if flow_dataset.empty:
            self.logger.error("FlowStatsfile.csv is empty or corrupt!")
            return

        try:
            if 'label' not in flow_dataset.columns:
                self.logger.error("Missing 'label' column in dataset.")
                return

            # Keep only selected feature columns
            X_flow = flow_dataset[self.feature_columns]
            X_flow = X_flow.astype('float64')
            y_flow = self.label_encoder.fit_transform(flow_dataset['label'])

            if len(X_flow) == 0 or len(y_flow) == 0:
                self.logger.error("No data to train the model. Please check FlowStatsfile.csv.")
                return

            X_train, X_test, y_train, y_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=42)
            classifier = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)
            self.flow_model = classifier.fit(X_train, y_train)
            y_pred = self.flow_model.predict(X_test)

            cm = confusion_matrix(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)

            self.logger.info("------------------------------------------------------------------------------")
            self.logger.info("Confusion matrix:\n{}".format(cm))
            self.logger.info("Success accuracy = {:.2f}%".format(acc * 100))
            self.logger.info("Fail accuracy = {:.2f}%".format((1 - acc) * 100))
            self.logger.info("------------------------------------------------------------------------------")

        except Exception as e:
            self.logger.error(f"Training error: {e}")

    def flow_predict(self):
        try:
            if not os.path.exists("PredictFlowStatsfile.csv"):
                self.logger.warning("PredictFlowStatsfile.csv not found.")
                return

            predict_flow_dataset = pd.read_csv("PredictFlowStatsfile.csv")
            if predict_flow_dataset.empty:
                self.logger.warning("PredictFlowStatsfile.csv is empty.")
                return

            X_predict_flow = predict_flow_dataset[self.feature_columns].astype('float64')
            if X_predict_flow.shape[0] == 0:
                self.logger.warning("No prediction rows.")
                return

            y_pred = self.flow_model.predict(X_predict_flow)
            labels = self.label_encoder.inverse_transform(y_pred)

            legitimate = sum(1 for label in labels if label == "BENIGN")
            ddos = len(labels) - legitimate

            self.logger.info("------------------------------------------------------------------------------")
            if legitimate / len(labels) * 100 > 80:
                self.logger.info("Legitimate traffic detected.")
            else:
                self.logger.info("DDoS traffic detected.")
                for idx, label in enumerate(labels):
                    if label != "BENIGN":
                        victim = int(predict_flow_dataset.iloc[idx, 5]) % 20
                        self.logger.info("Potential victim: host h{}".format(victim))
            self.logger.info("------------------------------------------------------------------------------")

            # Clear the prediction file
            with open("PredictFlowStatsfile.csv", "w") as file0:
                file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')

        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
    # ---------------------------- SECURITY ENHANCEMENTS ----------------------------
    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        self.label_encoder = LabelEncoder()
        self.flow_model = None
        self.blacklist = set()
        self.attack_log_file = "attack_log.txt"
        self.model_reload_interval = 300  # Retrain every 5 minutes
        hub.spawn(self._periodic_retrain)

        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        print("Training time:", (end - start))

    def _periodic_retrain(self):
        while True:
            hub.sleep(self.model_reload_interval)
            self.logger.info("🔄 Retraining model for updated patterns...")
            self.flow_training()

    def _log_ddos_attack(self, ip_dst, score=None):
        with open(self.attack_log_file, "a") as log:
            timestamp = datetime.now().isoformat()
            log.write(f"{timestamp} - Detected DDoS targeting {ip_dst} | Score: {score}\n")

    def _blacklist_host(self, ip):
        self.blacklist.add(ip)
        self.logger.info(f"🚫 IP {ip} has been blacklisted due to suspicious activity.")

    def _export_results(self, y_true, y_pred, name="results.csv"):
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred
        })
        df.to_csv(name, index=False)
        self.logger.info(f"🧾 Results exported to {name}.")

    # ---------------------- EXISTING SYSTEM EVALUATION ----------------------------
    def _baseline_detection_accuracy(self):
        """
        Compares ensemble classifier to basic packet rate thresholding.
        """
        self.logger.info("⚙️ Evaluating existing basic detection approach...")
        dataset = pd.read_csv("FlowStatsfile.csv")
        threshold = 5000  # arbitrary threshold for packet rate

        try:
            if 'label' not in dataset.columns:
                self.logger.warning("FlowStatsfile.csv missing 'label' for comparison.")
                return

            y_true = dataset['label']
            y_pred = ['DDoS' if p > threshold else 'BENIGN' for p in dataset['packet_count_per_second']]
            acc = accuracy_score(y_true, y_pred)
            self.logger.info(f"⚖️ Baseline Detection Accuracy: {acc * 100:.2f}%")
        except Exception as e:
            self.logger.error(f"Baseline detection error: {e}")
