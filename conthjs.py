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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2

ATTACK_EXPLANATION_FILE = 'suspected_attacks_explained.txt'
ATTACK_COMMANDS_FILE = 'ddos_attack_commands.txt'
TRAINING_DIR = 'xbsg'

attack_emoji_map = {
    'ldap': '📡',
    'mssql': '🛢️',
    'ntp': '⏱️',
    'snmp': '📶',
    'dns': '🌐',
    'netbios': '🧬',
    'portmap': '🔌',
}

class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        self.attack_explanations = self.load_attack_metadata()

        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        print("Training time: ", (end-start))

    def load_attack_metadata(self):
        try:
            with open(ATTACK_EXPLANATION_FILE, 'r') as file:
                data = file.read().split("\n\n")
                attack_map = {}
                for entry in data:
                    lines = entry.strip().split("\n")
                    if len(lines) > 1:
                        attack_name = lines[0].strip()
                        explanation = "\n".join(lines[1:])
                        attack_map[attack_name.lower()] = explanation
                return attack_map
        except Exception as e:
            self.logger.info(f"Error loading attack metadata: {e}")
            return {}

    def preprocess_data(self, df):
        df.fillna(0, inplace=True)
        return df.select_dtypes(include=['number']).astype('float64')

    def select_features(self, X, y):
        selector1 = SelectKBest(score_func=chi2, k=10).fit(X, y)
        selector2 = SelectKBest(score_func=f_classif, k=10).fit(X, y)
        selector3 = ExtraTreesClassifier(n_estimators=100).fit(X, y)

        features_1 = selector1.get_support(indices=True)
        features_2 = selector2.get_support(indices=True)
        features_3 = selector3.feature_importances_.argsort()[-10:]

        final_indices = list(set(features_1) | set(features_2) | set(features_3))
        return X[:, final_indices]

    def build_stacking_model(self):
        base_learners = [
            ('rf', RandomForestClassifier(n_estimators=100)),
            ('knn', KNeighborsClassifier(n_neighbors=3)),
            ('dt', DecisionTreeClassifier()),
            ('gb', GradientBoostingClassifier())
        ]
        meta_learner = LogisticRegression(max_iter=200)
        return StackingClassifier(estimators=base_learners, final_estimator=meta_learner)

    def handle_error(self, e, context=""):
        self.logger.info(f"❌ Error in {context}: {e}")
        self.logger.info("🔍 Why it happened: %s", type(e).__name__)
        if isinstance(e, FileNotFoundError):
            self.logger.info("💡 Fix: Make sure the dataset file exists at the path.")
        elif isinstance(e, ValueError):
            self.logger.info("💡 Fix: Check if the dataset has the expected number of features.")
        elif isinstance(e, ImportError):
            self.logger.info("💡 Fix: Install missing module using pip. Try: pip install -r requirements.txt")

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
            file0.write('timestamp,datapath_id,...\n')

    def flow_training(self):
        try:
            self.logger.info("⚙️ Starting training from xbsg directory...")
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

                    y_pred = model.predict(X_test)

                    key = file.lower().split("-")[0]
                    emoji = attack_emoji_map.get(key, '⚠️')

                    self.logger.info(f"{emoji} Model trained for {key.upper()} attack")
                    self.logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
                    self.logger.info("Accuracy: %.2f%%", accuracy_score(y_test, y_pred)*100)
                    self.logger.info("Precision: %.2f%%", precision_score(y_test, y_pred, average='macro')*100)
                    self.logger.info("Recall: %.2f%%", recall_score(y_test, y_pred, average='macro')*100)
                    self.logger.info("F1-Score: %.2f%%", f1_score(y_test, y_pred, average='macro')*100)

            self.logger.info("✅ Training completed for all datasets.")
        except Exception as e:
            self.handle_error(e, context="flow_training")

    def send_email_notification(self, subject, message):
        sender = "0felistus0@gmail.com"
        password = "@Taliah66"
        receiver = "0felistus0@gmail.com"

        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = receiver
        msg['Subject'] = subject

        msg.attach(MIMEText(message, 'plain'))

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
            server.quit()
        except Exception as e:
            self.logger.info(f"Email failed: {e}")

    def flow_predict(self):
        try:
            df = pd.read_csv('PredictFlowStatsfile.csv')
            df = self.preprocess_data(df)
            X = df.values
            y_pred = self.flow_model.predict(X)

            legitimate = sum(1 for i in y_pred if i == 0)
            ddos = len(y_pred) - legitimate

            self.logger.info("Detected Traffic Summary: %d legitimate | %d DDoS", legitimate, ddos)

            if ddos > legitimate:
                self.logger.info("DDoS Traffic Detected")
                victim = int(df.iloc[0, 5]) % 20
                self.logger.info("Victim is host: h%s", victim)
                self.send_email_notification("DDoS Alert", f"Potential DDoS detected targeting host h{victim}.")
                for cmd in open(ATTACK_COMMANDS_FILE):
                    attack_key = cmd.strip().lower()
                    if attack_key in self.attack_explanations:
                        self.logger.info("Attack Explanation: \n%s", self.attack_explanations[attack_key])
            else:
                self.logger.info("Legitimate Traffic")

            open("PredictFlowStatsfile.csv", "w").write('timestamp,datapath_id,...\n')

        except Exception as e:
            self.handle_error(e, context="flow_predict")
