# FedIDS - 3 Distinct Intrusion Datasets → 1 Global Model

This repository demonstrates a federated learning pipeline where **three different intrusion datasets**
(simulated here) are used by three clients to train local models. Each client sends **noisy** model updates
to the server. The server performs **FedAvg** aggregation and produces **one global model**.

**Important:** The datasets included here are **synthetic simulations** that mimic different dataset
characteristics so you can run the whole pipeline immediately. Replace the CSVs in `data/` with real
datasets (e.g., NSL-KDD, CICIDS2017, UNSW-NB15) if you want to run on real data — instructions below.

## Files
- `configs/config.json` - experiment configuration
- `data/datasetA.csv`, `data/datasetB.csv`, `data/datasetC.csv` - synthetic datasets (different distributions)
- `fed/` - federated code: `client.py`, `server.py`, `model.py`
- `run_federated.py` - run the federated training
- `evaluate_global.py` - evaluate the final global model on combined holdout
- `tools/standardize_csv.py` - utility to standardize CSV numeric columns
- `artifacts/` - global model weights saved here after training

## How to run
```bash
pip install -r requirements.txt
python run_federated.py
python evaluate_global.py
```

## Replace synthetic data with real datasets
1. Put your CSVs at `data/datasetA.csv`, `data/datasetB.csv`, `data/datasetC.csv`.
2. Ensure each CSV has the same target column name (default: `label`) and labels belong to the same set `['normal', 'dos', 'probe', 'r2l']`.
3. If features differ between datasets, preprocess them to have the same numeric feature columns and order (use `tools/standardize_csv.py` to z-score numeric columns).

## Notes
- Each client adds Gaussian noise to the weights before sending them to the server (`noise_std` in `configs/config.json`).
- FedAvg is weighted by the local number of samples per client.
- You can change the number of rounds, learning rate, local epochs in `configs/config.json`.
