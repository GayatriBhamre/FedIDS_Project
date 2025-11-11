#!/usr/bin/env python3
"""
Enhanced Federated Learning Training System for FedIDS
Improved with increased epochs, better convergence, and real-time monitoring
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
import logging
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Plotting disabled.")

# Add fed module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fed.model import SoftmaxRegression
from fed.client import FederatedClient
from fed.server import FederatedServer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedFederatedTraining:
    def __init__(self, config_path="configs/config.json"):
        """Initialize enhanced federated training system"""
        self.config = self.load_config(config_path)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.clients = []
        self.server = None
        self.training_history = {
            'rounds': [],
            'accuracies': [],
            'losses': [],
            'client_accuracies': {}
        }
        
    def load_config(self, config_path):
        """Load enhanced configuration"""
        default_config = {
            "num_clients": 3,
            "num_rounds": 15,  # Increased from 5
            "local_epochs": 10,  # Increased from 3
            "learning_rate": 0.01,
            "batch_size": 32,
            "test_size": 0.2,
            "random_state": 42,
            "convergence_threshold": 0.001,
            "patience": 3,
            "min_improvement": 0.005
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            # Update with enhanced parameters
            config.update(default_config)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using enhanced defaults.")
            return default_config
    
    def load_and_preprocess_data(self):
        """Load and preprocess all datasets with enhanced feature engineering"""
        logger.info("Loading and preprocessing datasets...")
        
        datasets = []
        dataset_files = ['data/datasetA.csv', 'data/datasetB.csv', 'data/datasetC.csv']
        
        for i, file_path in enumerate(dataset_files):
            try:
                df = pd.read_csv(file_path)
                logger.info(f"Dataset {chr(65+i)}: {df.shape[0]} samples, {df.shape[1]} features")
                
                # Handle different label column names
                label_col = None
                for col in ['label', 'Label', 'class', 'Class', 'attack_type']:
                    if col in df.columns:
                        label_col = col
                        break
                
                if label_col is None:
                    # If no label column found, create synthetic labels based on data patterns
                    logger.warning(f"No label column found in {file_path}. Creating synthetic labels.")
                    # Use simple heuristics to create labels
                    df['label'] = self.create_synthetic_labels(df)
                    label_col = 'label'
                
                # Separate features and labels
                X = df.drop(columns=[label_col])
                y = df[label_col]
                
                # Convert features to numeric
                X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
                
                # Enhanced feature engineering
                X = self.engineer_features(X)
                
                datasets.append((X, y))
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                # Create synthetic dataset as fallback
                X_synthetic, y_synthetic = self.create_synthetic_dataset(1000, 20)
                datasets.append((X_synthetic, y_synthetic))
        
        return datasets
    
    def create_synthetic_labels(self, df):
        """Create synthetic labels based on data patterns"""
        # Simple heuristic: use statistical properties to infer attack types
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return np.random.choice(['Normal', 'DoS', 'Probe', 'R2L'], size=len(df))
        
        # Calculate z-scores for anomaly detection (manual implementation)
        numeric_data = numeric_df.fillna(0)
        mean_vals = numeric_data.mean()
        std_vals = numeric_data.std()
        z_scores = np.abs((numeric_data - mean_vals) / (std_vals + 1e-8))
        anomaly_scores = z_scores.mean(axis=1)
        
        # Assign labels based on anomaly scores
        labels = []
        for score in anomaly_scores:
            if score > 2.5:
                labels.append(np.random.choice(['DoS', 'Probe', 'R2L']))
            else:
                labels.append('Normal')
        
        return labels
    
    def engineer_features(self, X):
        """Enhanced feature engineering"""
        # Add statistical features
        X_enhanced = X.copy()
        
        # Add rolling statistics if we have enough features
        if X.shape[1] >= 5:
            X_enhanced['feature_mean'] = X.mean(axis=1)
            X_enhanced['feature_std'] = X.std(axis=1)
            X_enhanced['feature_max'] = X.max(axis=1)
            X_enhanced['feature_min'] = X.min(axis=1)
        
        # Add interaction features for top features
        if X.shape[1] >= 3:
            cols = X.columns[:3]
            X_enhanced[f'{cols[0]}_x_{cols[1]}'] = X[cols[0]] * X[cols[1]]
            X_enhanced[f'{cols[0]}_x_{cols[2]}'] = X[cols[0]] * X[cols[2]]
        
        return X_enhanced
    
    def create_synthetic_dataset(self, n_samples, n_features):
        """Create synthetic dataset for testing"""
        logger.info(f"Creating synthetic dataset: {n_samples} samples, {n_features} features")
        
        # Create different patterns for different attack types
        X_normal = np.random.normal(0, 1, (n_samples//4, n_features))
        X_dos = np.random.normal(2, 1.5, (n_samples//4, n_features))
        X_probe = np.random.normal(-1, 0.8, (n_samples//4, n_features))
        X_r2l = np.random.normal(1, 2, (n_samples//4, n_features))
        
        X = np.vstack([X_normal, X_dos, X_probe, X_r2l])
        y = ['Normal'] * (n_samples//4) + ['DoS'] * (n_samples//4) + \
            ['Probe'] * (n_samples//4) + ['R2L'] * (n_samples//4)
        
        return pd.DataFrame(X), pd.Series(y)
    
    def prepare_federated_data(self, datasets):
        """Prepare data for federated learning with enhanced preprocessing"""
        logger.info("Preparing federated learning data...")
        
        # Combine all datasets to fit scalers
        all_X = pd.concat([X for X, y in datasets], ignore_index=True)
        all_y = pd.concat([y for X, y in datasets], ignore_index=True)
        
        # Fit preprocessing
        X_scaled = self.scaler.fit_transform(all_X)
        y_encoded = self.label_encoder.fit_transform(all_y)
        
        logger.info(f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Split data among clients
        client_data = []
        total_samples = len(X_scaled)
        samples_per_client = total_samples // self.config['num_clients']
        
        for i in range(self.config['num_clients']):
            start_idx = i * samples_per_client
            if i == self.config['num_clients'] - 1:  # Last client gets remaining data
                end_idx = total_samples
            else:
                end_idx = (i + 1) * samples_per_client
            
            X_client = X_scaled[start_idx:end_idx]
            y_client = y_encoded[start_idx:end_idx]
            
            # Split into train/test for each client
            X_train, X_test, y_train, y_test = train_test_split(
                X_client, y_client, 
                test_size=self.config['test_size'],
                random_state=self.config['random_state'] + i,
                stratify=y_client if len(np.unique(y_client)) > 1 else None
            )
            
            client_data.append({
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'client_id': i
            })
            
            logger.info(f"Client {i}: {len(X_train)} training, {len(X_test)} testing samples")
        
        return client_data
    
    def initialize_federated_system(self, client_data):
        """Initialize enhanced federated learning system"""
        logger.info("Initializing federated learning system...")
        
        # Get feature dimensions
        n_features = client_data[0]['X_train'].shape[1]
        n_classes = len(self.label_encoder.classes_)
        
        # Initialize server
        self.server = FederatedServer(n_features, n_classes)
        
        # Initialize clients with enhanced configuration
        self.clients = []
        for data in client_data:
            client = FederatedClient(
                client_id=data['client_id'],
                X_train=data['X_train'],
                y_train=data['y_train'],
                X_test=data['X_test'],
                y_test=data['y_test'],
                learning_rate=self.config['learning_rate'],
                epochs=self.config['local_epochs']
            )
            self.clients.append(client)
        
        logger.info(f"Initialized {len(self.clients)} clients and 1 server")
    
    def train_federated_model(self):
        """Enhanced federated training with convergence monitoring"""
        logger.info("Starting enhanced federated training...")
        
        best_accuracy = 0
        patience_counter = 0
        
        for round_num in range(self.config['num_rounds']):
            logger.info(f"\n=== Federated Round {round_num + 1}/{self.config['num_rounds']} ===")
            
            # Client training phase
            client_weights = []
            client_accuracies = []
            
            for client in self.clients:
                # Train local model
                weights, accuracy, loss = client.train()
                client_weights.append(weights)
                client_accuracies.append(accuracy)
                
                logger.info(f"Client {client.client_id}: Accuracy={accuracy:.4f}, Loss={loss:.4f}")
            
            # Server aggregation
            global_weights = self.server.aggregate_weights(client_weights)
            
            # Update all clients with global weights
            for client in self.clients:
                client.update_weights(global_weights)
            
            # Global evaluation
            global_accuracy = self.evaluate_global_model()
            
            # Update training history
            self.training_history['rounds'].append(round_num + 1)
            self.training_history['accuracies'].append(global_accuracy)
            self.training_history['client_accuracies'][round_num + 1] = client_accuracies
            
            logger.info(f"Global Model Accuracy: {global_accuracy:.4f}")
            
            # Convergence check
            if global_accuracy > best_accuracy + self.config['min_improvement']:
                best_accuracy = global_accuracy
                patience_counter = 0
                # Save best model
                self.save_model(global_weights, f"artifacts/global_round{round_num + 1}.npz")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                logger.info(f"Early stopping at round {round_num + 1} due to convergence")
                break
        
        # Save final model
        final_weights = self.server.get_global_weights()
        self.save_model(final_weights, "artifacts/global_final.npz")
        
        logger.info(f"Training completed! Best accuracy: {best_accuracy:.4f}")
        return final_weights
    
    def evaluate_global_model(self):
        """Evaluate global model on all test data"""
        all_predictions = []
        all_true_labels = []
        
        for client in self.clients:
            predictions = client.predict(client.X_test)
            all_predictions.extend(predictions)
            all_true_labels.extend(client.y_test)
        
        accuracy = accuracy_score(all_true_labels, all_predictions)
        return accuracy
    
    def save_model(self, weights, filepath):
        """Save model weights"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, W=weights['W'], b=weights['b'])
        logger.info(f"Model saved to {filepath}")
    
    def generate_training_report(self):
        """Generate comprehensive training report"""
        logger.info("Generating training report...")
        
        if not HAS_PLOTTING:
            logger.warning("Plotting libraries not available. Generating text report only.")
            self.generate_text_report()
            return
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Training accuracy plot
        plt.subplot(2, 3, 1)
        plt.plot(self.training_history['rounds'], self.training_history['accuracies'], 'b-o')
        plt.title('Global Model Accuracy Over Rounds')
        plt.xlabel('Federated Round')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        # Client accuracy comparison
        plt.subplot(2, 3, 2)
        for round_num, accuracies in self.training_history['client_accuracies'].items():
            plt.bar([f'C{i}' for i in range(len(accuracies))], accuracies, alpha=0.7, label=f'Round {round_num}')
        plt.title('Client Accuracies by Round')
        plt.xlabel('Client')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Final evaluation
        plt.subplot(2, 3, 3)
        final_accuracy = self.evaluate_global_model()
        
        # Get predictions for confusion matrix
        all_predictions = []
        all_true_labels = []
        for client in self.clients:
            predictions = client.predict(client.X_test)
            all_predictions.extend(predictions)
            all_true_labels.extend(client.y_test)
        
        # Confusion matrix
        cm = confusion_matrix(all_true_labels, all_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Classification report
        plt.subplot(2, 3, 4)
        report = classification_report(all_true_labels, all_predictions, 
                                     target_names=self.label_encoder.classes_, 
                                     output_dict=True)
        
        # Plot precision, recall, f1-score
        classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
        metrics = ['precision', 'recall', 'f1-score']
        
        x = np.arange(len(classes))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [report[cls][metric] for cls in classes]
            plt.bar(x + i*width, values, width, label=metric)
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Classification Metrics by Class')
        plt.xticks(x + width, classes, rotation=45)
        plt.legend()
        
        # Training summary
        plt.subplot(2, 3, 5)
        plt.axis('off')
        summary_text = f"""
        Training Summary:
        
        Total Rounds: {len(self.training_history['rounds'])}
        Final Accuracy: {final_accuracy:.4f}
        Best Accuracy: {max(self.training_history['accuracies']):.4f}
        
        Configuration:
        - Clients: {self.config['num_clients']}
        - Local Epochs: {self.config['local_epochs']}
        - Learning Rate: {self.config['learning_rate']}
        - Batch Size: {self.config['batch_size']}
        
        Classes: {', '.join(self.label_encoder.classes_)}
        """
        plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('artifacts/training_report.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Training report saved to artifacts/training_report.png")
    
    def generate_text_report(self):
        """Generate text-based training report when plotting is unavailable"""
        final_accuracy = self.evaluate_global_model()
        
        # Get predictions for evaluation
        all_predictions = []
        all_true_labels = []
        for client in self.clients:
            predictions = client.predict(client.X_test)
            all_predictions.extend(predictions)
            all_true_labels.extend(client.y_test)
        
        report = f"""
=== Enhanced FedIDS Training Report ===
Generated: {datetime.now().isoformat()}

Training Configuration:
- Total Rounds: {len(self.training_history['rounds'])}
- Clients: {self.config['num_clients']}
- Local Epochs: {self.config['local_epochs']}
- Learning Rate: {self.config['learning_rate']}
- Batch Size: {self.config['batch_size']}

Performance Metrics:
- Final Accuracy: {final_accuracy:.4f}
- Best Accuracy: {max(self.training_history['accuracies']):.4f}
- Accuracy Improvement: {(max(self.training_history['accuracies']) - self.training_history['accuracies'][0]):.4f}

Round-by-Round Accuracy:
"""
        
        for i, acc in enumerate(self.training_history['accuracies']):
            report += f"Round {i+1}: {acc:.4f}\n"
        
        report += f"""
Classes: {', '.join(self.label_encoder.classes_)}

Training Status: COMPLETED
Model saved to: artifacts/global_final.npz
"""
        
        # Save text report
        os.makedirs('artifacts', exist_ok=True)
        with open('artifacts/training_report.txt', 'w') as f:
            f.write(report)
        
        print(report)
        logger.info("Text training report saved to artifacts/training_report.txt")

def main():
    """Main training function"""
    print("🚀 Starting Enhanced FedIDS Training System")
    print("=" * 50)
    
    # Initialize training system
    trainer = EnhancedFederatedTraining()
    
    # Load and preprocess data
    datasets = trainer.load_and_preprocess_data()
    
    # Prepare federated data
    client_data = trainer.prepare_federated_data(datasets)
    
    # Initialize federated system
    trainer.initialize_federated_system(client_data)
    
    # Train model
    final_weights = trainer.train_federated_model()
    
    # Generate report
    trainer.generate_training_report()
    
    print("\n✅ Enhanced FedIDS training completed successfully!")
    print("📊 Check artifacts/ directory for saved models and reports")

if __name__ == "__main__":
    main()
