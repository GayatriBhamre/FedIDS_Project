import os, json, numpy as np

class Server:
    def __init__(self, n_features, n_classes, artifacts_dir="artifacts"):
        self.n_features = n_features
        self.n_classes = n_classes
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        rng = np.random.default_rng(42)
        self.global_weights = {"W": rng.normal(0, 0.01, size=(n_features, n_classes)), "b": np.zeros(n_classes, dtype=float)}
        self.round = 0

    def get_global_weights(self):
        return self.global_weights

    def aggregate(self, client_payloads):
        total = sum(p["num_samples"] for p in client_payloads)
        W_acc = np.zeros_like(self.global_weights["W"])
        b_acc = np.zeros_like(self.global_weights["b"])
        for p in client_payloads:
            w = p["weights"]; n = p["num_samples"]
            W_acc += (n / total) * w["W"]
            b_acc += (n / total) * w["b"]
        self.global_weights = {"W": W_acc, "b": b_acc}
        self.round += 1
        np_path = os.path.join(self.artifacts_dir, f"global_round{self.round}.npz")
        import numpy as _np
        _np.savez(np_path, W=self.global_weights["W"], b=self.global_weights["b"])
        return self.global_weights

    def save_final(self):
        import numpy as _np, os
        final = os.path.join(self.artifacts_dir, "global_final.npz")
        _np.savez(final, W=self.global_weights["W"], b=self.global_weights["b"])
        return final
