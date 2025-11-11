import numpy as np

class SoftmaxRegression:
    def __init__(self, n_features, n_classes, lr=0.05, l2=1e-4, batch_size=128, seed=42):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr
        self.l2 = l2
        self.batch_size = batch_size
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, size=(n_features, n_classes))
        self.b = np.zeros(n_classes, dtype=float)

    def get_weights(self):
        return {"W": self.W.copy(), "b": self.b.copy()}

    def set_weights(self, weights):
        self.W = weights["W"].copy()
        self.b = weights["b"].copy()

    @staticmethod
    def _softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    @staticmethod
    def _one_hot(y, n_classes):
        yh = np.zeros((len(y), n_classes), dtype=float)
        yh[np.arange(len(y)), y] = 1.0
        return yh

    def _forward(self, X):
        logits = X @ self.W + self.b
        probs = self._softmax(logits)
        return probs

    def predict(self, X):
        probs = self._forward(X)
        return np.argmax(probs, axis=1)

    def fit_one_epoch(self, X, y, seed=42):
        n = X.shape[0]
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            bidx = idx[start:end]
            Xb = X[bidx]; yb = y[bidx]
            probs = self._forward(Xb)
            yh = self._one_hot(yb, self.n_classes)
            grad_logits = (probs - yh) / Xb.shape[0]
            gW = Xb.T @ grad_logits + self.l2 * self.W
            gb = np.sum(grad_logits, axis=0)
            self.W -= self.lr * gW
            self.b -= self.lr * gb
