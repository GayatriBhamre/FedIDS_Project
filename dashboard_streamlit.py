# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import shap

# st.set_page_config(page_title="Federated IDS Dashboard", layout="wide")

# # ------------------------------
# # Load global model (from npz)
# # ------------------------------
# def load_model():
#     try:
#         data = np.load("artifacts/global_final.npz", allow_pickle=True)
#         st.sidebar.write("Model keys found:", list(data.keys()))  # debug info

#         # Take first available arrays (W, b, etc.)
#         model = {key: data[key] for key in data.files}
#         return model
#     except Exception as e:
#         st.error(f"Intrusion detection not available yet. Error: {e}")
#         return None


# # ------------------------------
# # Prediction Function (Dummy Example)
# # ------------------------------
# def predict_intrusion(model, df):
#     np.random.seed(42)

#     # Try to use weights shape to make predictions more realistic
#     if "W" in model and df.shape[1] >= model["W"].shape[0]:
#         X = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
#         W = model["W"]
#         preds_raw = X @ W  # dot product
#         preds = np.argmax(preds_raw, axis=1)
#         labels = ["Normal", "DoS", "Probe", "R2L"]
#         preds = [labels[p % len(labels)] for p in preds]
#     else:
#         # fallback: random predictions
#         preds = np.random.choice(["Normal", "Attack"], size=len(df))

#     return preds


# # ------------------------------
# # Streamlit UI
# # ------------------------------
# st.title("🛡️ Federated Intrusion Detection System Dashboard")

# uploaded_file = st.file_uploader("📂 Upload Network Traffic CSV", type=["csv"])

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.subheader("📊 Uploaded Data Preview")
#     st.write(df.head())

#     model = load_model()

#     if model is not None:
#         st.subheader("🚨 Intrusion Detection Results")
#         preds = predict_intrusion(model, df)

#         df["Prediction"] = preds
#         st.write(df.head())

#         # ------------------------------
#         # Bar Graph of Predictions
#         # ------------------------------
#         st.subheader("📉 Prediction Distribution")
#         pred_counts = df["Prediction"].value_counts()

#         fig, ax = plt.subplots()
#         pred_counts.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c", "#3498db", "#f39c12"])
#         ax.set_ylabel("Count")
#         ax.set_title("Intrusion Detection Results")
#         st.pyplot(fig)

#         # ------------------------------
#         # Explainability with SHAP
#         # ------------------------------
#         st.subheader("🧠 Model Explainability (SHAP)")
#         st.info("Showing SHAP values for first 100 samples (demo).")

#         try:
#             X_sample = df.drop(columns=["Prediction"]).select_dtypes(include=[np.number]).head(100)

#             if not X_sample.empty:
#                 explainer = shap.Explainer(lambda x: np.random.randn(x.shape[0], len(set(preds))), X_sample)
#                 shap_values = explainer(X_sample)

#                 st.set_option('deprecation.showPyplotGlobalUse', False)
#                 shap.summary_plot(shap_values, X_sample, show=False)
#                 st.pyplot(bbox_inches="tight")
#         except Exception as e:
#             st.error(f"SHAP explainability failed: {e}")

# else:
#     st.info(" Upload a CSV file to start intrusion detection.") 

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="FedIDS Dashboard", layout="wide")

st.title("🛡️Intrusion Detection System Dashboard")

# ----------------- Load Global Model -----------------
try:
    model_data = np.load("artifacts/global_final.npz", allow_pickle=True)
    W = model_data['W']
    b = model_data['b']
    st.success("Global model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load global model: {e}")
    st.stop()

# ----------------- File Upload -----------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        # Drop non-feature columns
        if "Label" in data.columns:
            X = data.drop(columns=["Label"]).values
        else:
            X = data.iloc[:, :20].values  # first 20 columns as features

        # ----------------- Predictions -----------------
        logits = np.dot(X, W) + b
        predictions = np.argmax(logits, axis=1)

        # Map numeric predictions to attack types
        attack_mapping = {0: "Normal", 1: "DoS", 2: "PortScan", 3: "BruteForce"}
        pred_labels = [attack_mapping.get(p, "Unknown") for p in predictions]

        # ----------------- Display Results -----------------
        st.subheader("Predictions")
        st.write(pd.DataFrame({"Prediction": pred_labels}))

        # ----------------- Bar Chart -----------------
        st.subheader("Prediction Counts")
        pred_counts = pd.Series(pred_labels).value_counts()
        fig, ax = plt.subplots(figsize=(3,2))  # small figure
        pred_counts.plot(kind='barh', ax=ax, color='skyblue')  # horizontal bars
        ax.set_xlabel("Count")
        ax.set_ylabel("Attack Type")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing file: {e}")

# ----------------- Footer -----------------
st.markdown(
    "<hr><p style='text-align:center; color:gray;'>Intrusion Detection System using Federated Learning</p>",
    unsafe_allow_html=True
)
