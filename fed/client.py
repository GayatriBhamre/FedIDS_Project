import json, os, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from .model import SoftmaxRegression

def add_gaussian_noise_to_weights(weights, std=0.0, seed=None):
    if std <= 0:
        return {"W": weights["W"].copy(), "b": weights["b"].copy()}
    rng = np.random.default_rng(seed)
    return {"W": weights["W"] + rng.normal(0, std, size=weights["W"].shape),
            "b": weights["b"] + rng.normal(0, std, size=weights["b"].shape)}

class Client:
    def __init__(self, cid, data_path, config_path="configs/config.json"):
        self.cid = cid
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.data_path = data_path
        self.target = self.cfg["target_col"]
        self.class_names = self.cfg["class_names"]
        self.lr = self.cfg["learning_rate"]
        self.l2 = self.cfg["l2_reg"]
        self.batch_size = self.cfg["batch_size"]
        self.local_epochs = self.cfg["local_epochs"]
        self.noise_std = self.cfg["noise_std"]
        self.train_split = self.cfg["train_split"]
        self._load_data()

    def _load_data(self):
        df = pd.read_csv(self.data_path)
        assert self.target in df.columns, f"Target '{self.target}' missing in {self.data_path}"
        X = df.drop(columns=[self.target]).apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y_raw = df[self.target].astype(str).values
        le = LabelEncoder(); le.fit(self.class_names)
        y = le.transform(y_raw)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=self.train_split, random_state=42, stratify=y)
        self.n_features = self.X_train.shape[1]
        self.n_classes = len(self.class_names)

    def init_model(self, global_weights=None):
        self.model = SoftmaxRegression(n_features=self.n_features, n_classes=self.n_classes, lr=self.lr, l2=self.l2, batch_size=self.batch_size, seed=42+self.cid)
        if global_weights is not None:
            self.model.set_weights(global_weights)

    def local_train(self, global_weights=None):
        self.init_model(global_weights)
        for ep in range(self.local_epochs):
            self.model.fit_one_epoch(self.X_train, self.y_train, seed=1000 + self.cid + ep)
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, target_names=self.class_names, zero_division=0)
        weights = self.model.get_weights()
        noisy = add_gaussian_noise_to_weights(weights, std=self.noise_std, seed=777 + self.cid)
        payload = {"client_id": self.cid, "num_samples": int(self.X_train.shape[0]), "weights": noisy, "metrics": {"local_test_acc": float(acc)}}
        return payload, report
