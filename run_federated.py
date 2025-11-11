import json, numpy as np
from fed.client import Client
from fed.server import Server
def main():
    with open("configs/config.json","r", encoding="utf-8") as f:
        cfg = json.load(f)
    np.random.seed(cfg["random_seed"])
    data_paths = cfg["data_paths"]
    clients = [Client(cid=i+1, data_path=dp, config_path="configs/config.json") for i, dp in enumerate(data_paths)]
    server = Server(clients[0].n_features, clients[0].n_classes, artifacts_dir=cfg["artifacts_dir"])
    for r in range(cfg["rounds"]):
        print(f"\\n=== Federated Round {r+1}/{cfg['rounds']} ===")
        gw = server.get_global_weights()
        payloads = []
        for c in clients:
            payload, report = c.local_train(global_weights=gw)
            payloads.append(payload)
            print(f"[Client {c.cid}] local_test_acc={payload['metrics']['local_test_acc']:.4f}")
        server.aggregate(payloads)
        print(f"[Server] Aggregated global weights for round {r+1} saved.")
    final = server.save_final()
    print(f"[Server] Final global model saved at: {final}")
if __name__ == "__main__": main()
