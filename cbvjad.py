# RF_controller.py - Fully Enhanced SDN Controller for DDoS Detection in IoT

from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub

import switch
from datetime import datetime
import pandas as pd
import os
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, jsonify, request
import threading

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2

# Files and directories
ATTACK_EXPLANATION_FILE = 'suspected_attacks_explained.txt'
ATTACK_COMMANDS_FILE = 'ddos_attack_commands.txt'
TRAINING_DIR = 'xbsg'
LOG_EXPORT_FILE = 'ddos_logs.json'

# Emoji map for attack labels
attack_emoji_map = {
    'ldap': '📡', 'mssql': '🛢️', 'ntp': '⏱️', 'snmp': '📶',
    'dns': '🌐', 'netbios': '🧬', 'portmap': '🔌', 'syn': '⚡', 'udp': '🌊'
}

# REST API App
app = Flask(__name__)
app_data = {'last_detection': {}, 'metrics': {}, 'logs': []}


class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        self.attack_explanations = self.load_attack_metadata()
        self.flow_model = None

        self.logger.info("⚙️ Initializing training...")
        try:
            self.flow_training()
        except Exception as e:
            self.handle_error(e, context="initial_training")

    def load_attack_metadata(self):
        try:
            with open(ATTACK_EXPLANATION_FILE, 'r') as file:
                data = file.read().split("\n\n")
                return {
                    lines[0].strip().lower(): "\n".join(lines[1:])
                    for entry in data if (lines := entry.strip().split("\n")) and len(lines) > 1
                }
        except Exception as e:
            self.logger.info(f"Error loading attack metadata: {e}")
            return {}

    def preprocess_data(self, df):
        df.fillna(0, inplace=True)
        if not pd.api.types.is_numeric_dtype(df.iloc[:, -1]):
            df.iloc[:, -1] = df.iloc[:, -1].astype(str).astype(int)
        return df.select_dtypes(include=['number']).astype('float64')

    def select_features(self, X, y):
        selector1 = SelectKBest(score_func=chi2, k=10).fit(X, y)
        selector2 = SelectKBest(score_func=f_classif, k=10).fit(X, y)
        selector3 = ExtraTreesClassifier(n_estimators=100).fit(X, y)
        return X[:, list(set(selector1.get_support(indices=True)) |
                         set(selector2.get_support(indices=True)) |
                         set(selector3.feature_importances_.argsort()[-10:]))]

    def build_stacking_model(self):
        base_learners = [
            ('rf', RandomForestClassifier(n_estimators=100)),
            ('knn', KNeighborsClassifier(n_neighbors=3)),
            ('dt', DecisionTreeClassifier()),
            ('gb', GradientBoostingClassifier())
        ]
        return StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression(max_iter=200))

    def handle_error(self, e, context=""):
        self.logger.info(f"❌ Error in {context}: {e}")
        self.logger.info("🔍 Why it happened: %s", type(e).__name__)
        if isinstance(e, FileNotFoundError):
            self.logger.info("💡 Fix: Make sure the dataset file exists.")
        elif isinstance(e, ValueError):
            self.logger.info("💡 Fix: Check dataset columns and data types.")

    def flow_training(self):
        logs = []
        for file in os.listdir(TRAINING_DIR):
            if file.endswith(".parquet") and "training" in file.lower():
                path = os.path.join(TRAINING_DIR, file)
                df = pd.read_parquet(path)
                df = self.preprocess_data(df)
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_selected = self.select_features(X_scaled, y)

                X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.25, random_state=0)
                model = self.build_stacking_model()
                model.fit(X_train, y_train)

                self.flow_model = model

                y_pred = model.predict(X_test)
                key = file.lower().split("-")[0]
                emoji = attack_emoji_map.get(key, '⚠️')

                log = {
                    "attack": key.upper(),
                    "emoji": emoji,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average='macro'),
                    "recall": recall_score(y_test, y_pred, average='macro'),
                    "f1": f1_score(y_test, y_pred, average='macro')
                }
                logs.append(log)
                self.logger.info(f"{emoji} {log['attack']} trained - Acc: {log['accuracy']*100:.2f}% F1: {log['f1']*100:.2f}%")
        app_data['metrics'] = logs
        with open(LOG_EXPORT_FILE, 'w') as f:
            json.dump(logs, f, indent=2)

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)
            self.flow_predict()

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[ev.datapath.id] = ev.datapath
        elif ev.state == DEAD_DISPATCHER:
            self.datapaths.pop(ev.datapath.id, None)

    def _request_stats(self, datapath):
        parser = datapath.ofproto_parser
        datapath.send_msg(parser.OFPFlowStatsRequest(datapath))

    def flow_predict(self):
        try:
            df = pd.read_csv("PredictFlowStatsfile.csv")
            df = self.preprocess_data(df)
            X = df.values
            y_pred = self.flow_model.predict(X)

            legitimate = sum(1 for i in y_pred if i == 0)
            ddos = len(y_pred) - legitimate
            app_data['last_detection'] = {'legitimate': legitimate, 'ddos': ddos}

            if ddos > legitimate:
                self.logger.info("DDoS Traffic Detected 🚨") 
                victim = int(df.iloc[0, 5]) % 20
                self.send_email_notification("DDoS Alert", f"Potential DDoS targeting host h{victim}.")
                for cmd in open(ATTACK_COMMANDS_FILE):
                    key = cmd.strip().lower()
                    if key in self.attack_explanations:
                        self.logger.info("📘 Explanation:\n%s", self.attack_explanations[key])
        except Exception as e:
            self.handle_error(e, context="flow_predict")

    def send_email_notification(self, subject, message):
        sender, password, receiver = "0felistus0@gmail.com", "@Taliah66", "0felistus0@gmail.com"
        msg = MIMEMultipart(); msg['From'] = sender; msg['To'] = receiver; msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587); server.starttls()
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
            server.quit()
        except Exception as e:
            self.logger.info(f"Email failed: {e}")


# REST API thread for real-time access
def start_rest_api():
    @app.route('/status', methods=['GET'])
    def status():
        return jsonify(app_data)

    @app.route('/metrics', methods=['GET'])
    def metrics():
        return jsonify(app_data.get('metrics', []))

    @app.route('/logs', methods=['GET'])
    def logs():
        if os.path.exists(LOG_EXPORT_FILE):
            with open(LOG_EXPORT_FILE) as f:
                return jsonify(json.load(f))
        return jsonify([])

    app.run(port=5001, debug=False, use_reloader=False)

# Start REST API in a separate thread
api_thread = threading.Thread(target=start_rest_api)
api_thread.setDaemon(True)
api_thread.start()
