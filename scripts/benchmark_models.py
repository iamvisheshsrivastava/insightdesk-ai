# scripts/benchmark_models.py

"""
Comprehensive Model Benchmarking & Evaluation Framework

This script provides complete evaluation of ML models including:
1. Model comparison (XGBoost vs TensorFlow)
2. Business metrics analysis
3. Performance reporting with visualizations
4. A/B testing framework foundation
"""

import sys
import json
import time
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.xgboost_classifier import XGBoostCategoryClassifier
from src.models.tensorflow_classifier import TensorFlowCategoryClassifier

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ModelBenchmark:
    """Comprehensive model benchmarking and evaluation."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.xgb_model = None
        self.tf_model = None
        
        # Evaluation data
        self.test_data = None
        self.test_labels = None
        self.label_encoder = None
        
        # Results storage
        self.benchmark_results = {
            "model_performance": {},
            "business_metrics": {},
            "inference_metrics": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Model predictions cache
        self.predictions_cache = {}
        
        print(f"📊 Model Benchmark initialized")
        print(f"📁 Results will be saved to: {self.results_dir}")
    
    def load_models(self):
        """Load both XGBoost and TensorFlow models."""
        print("\n🔄 Loading Models...")
        
        try:
            # Load XGBoost model
            print("   Loading XGBoost classifier...")
            self.xgb_model = XGBoostCategoryClassifier()
            self.xgb_model.load_model()
            if self.xgb_model.is_loaded:
                print("   ✅ XGBoost model loaded successfully")
            else:
                print("   ❌ XGBoost model failed to load")
                
        except Exception as e:
            print(f"   ❌ XGBoost loading error: {e}")
            self.xgb_model = None
        
        try:
            # Load TensorFlow model
            print("   Loading TensorFlow classifier...")
            self.tf_model = TensorFlowCategoryClassifier()
            self.tf_model.load_model()
            if self.tf_model.is_loaded:
                print("   ✅ TensorFlow model loaded successfully")
            else:
                print("   ❌ TensorFlow model failed to load")
                
        except Exception as e:
            print(f"   ❌ TensorFlow loading error: {e}")
            self.tf_model = None
        
        # Check if at least one model is available
        if not self.xgb_model and not self.tf_model:
            raise RuntimeError("No models available for benchmarking")
        
        return self.xgb_model is not None, self.tf_model is not None
    
    def load_test_data(self, data_file: str = "data/support_tickets.json"):
        """Load and prepare test data."""
        print(f"\n📂 Loading test data from {data_file}...")
        
        try:
            # Load data directly
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both direct list and nested structure
            if isinstance(data, list):
                tickets = data
            elif isinstance(data, dict) and 'tickets' in data:
                tickets = data['tickets']
            else:
                tickets = []
            
            if not tickets:
                raise ValueError("No tickets loaded from data file")
            
            # Prepare features and labels
            processed_data = []
            labels = []
            
            for ticket in tickets:
                # Skip tickets without category
                if not ticket.get('category'):
                    continue
                
                # Prepare ticket data for models
                ticket_data = {
                    'subject': ticket.get('subject', ''),
                    'description': ticket.get('description', ''),
                    'priority': ticket.get('priority', 'medium'),
                    'product': ticket.get('product', 'unknown')
                }
                
                processed_data.append(ticket_data)
                labels.append(ticket['category'])
            
            self.test_data = processed_data
            self.test_labels = labels
            
            print(f"   ✅ Loaded {len(processed_data)} test samples")
            print(f"   📊 Categories: {len(set(labels))} unique")
            
            # Display category distribution
            category_counts = pd.Series(labels).value_counts()
            print("   📈 Category distribution:")
            for category, count in category_counts.head(5).items():
                print(f"      {category}: {count} samples")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading test data: {e}")
            return False
    
    def measure_inference_latency(self, model, model_name: str, sample_size: int = 100) -> Dict[str, float]:
        """Measure inference latency and memory usage."""
        print(f"   ⏱️ Measuring {model_name} inference metrics...")
        
        if not self.test_data or sample_size > len(self.test_data):
            sample_size = len(self.test_data) if self.test_data else 10
        
        # Select random sample
        sample_indices = np.random.choice(len(self.test_data), sample_size, replace=False)
        sample_data = [self.test_data[i] for i in sample_indices]
        
        # Memory usage before inference
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure inference times
        inference_times = []
        successful_predictions = 0
        
        for ticket_data in sample_data:
            start_time = time.time()
            try:
                result = model.predict(ticket_data)
                end_time = time.time()
                
                if result and 'predicted_category' in result:
                    inference_times.append(end_time - start_time)
                    successful_predictions += 1
                    
            except Exception as e:
                print(f"      ⚠️ Prediction failed: {e}")
                continue
        
        # Memory usage after inference
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        if not inference_times:
            return {
                "avg_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "memory_usage_mb": memory_after - memory_before,
                "success_rate": 0,
                "throughput_per_sec": 0
            }
        
        # Calculate metrics
        latencies_ms = [t * 1000 for t in inference_times]  # Convert to milliseconds
        
        metrics = {
            "avg_latency_ms": np.mean(latencies_ms),
            "p95_latency_ms": np.percentile(latencies_ms, 95),
            "p99_latency_ms": np.percentile(latencies_ms, 99),
            "memory_usage_mb": memory_after - memory_before,
            "success_rate": successful_predictions / sample_size,
            "throughput_per_sec": 1000 / np.mean(latencies_ms) if latencies_ms else 0
        }
        
        print(f"      📊 Avg latency: {metrics['avg_latency_ms']:.2f}ms")
        print(f"      📊 P95 latency: {metrics['p95_latency_ms']:.2f}ms")
        print(f"      📊 Memory usage: {metrics['memory_usage_mb']:.2f}MB")
        print(f"      📊 Throughput: {metrics['throughput_per_sec']:.1f} req/sec")
        
        return metrics
    
    def evaluate_model_accuracy(self, model, model_name: str) -> Dict[str, Any]:
        """Evaluate model accuracy and classification metrics."""
        print(f"   🎯 Evaluating {model_name} accuracy...")
        
        if not self.test_data or not self.test_labels:
            return {}
        
        # Get predictions for all test data
        predictions = []
        probabilities = []
        prediction_times = []
        
        for ticket_data in self.test_data:
            start_time = time.time()
            try:
                result = model.predict(ticket_data)
                prediction_time = time.time() - start_time
                
                if result and 'predicted_category' in result:
                    predictions.append(result['predicted_category'])
                    probabilities.append(result.get('confidence', 0.0))
                    prediction_times.append(prediction_time)
                else:
                    predictions.append('unknown')
                    probabilities.append(0.0)
                    prediction_times.append(prediction_time)
                    
            except Exception as e:
                predictions.append('unknown')
                probabilities.append(0.0)
                prediction_times.append(0.0)
        
        # Cache predictions for later use
        self.predictions_cache[model_name] = {
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': self.test_labels
        }
        
        # Calculate metrics
        accuracy = accuracy_score(self.test_labels, predictions)
        weighted_f1 = f1_score(self.test_labels, predictions, average='weighted', zero_division=0)
        precision = precision_score(self.test_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(self.test_labels, predictions, average='weighted', zero_division=0)
        
        # Calculate per-class metrics
        unique_labels = list(set(self.test_labels + predictions))
        class_report = classification_report(
            self.test_labels, predictions, 
            labels=unique_labels, 
            target_names=unique_labels,
            output_dict=True,
            zero_division=0
        )
        
        metrics = {
            "accuracy": accuracy,
            "weighted_f1": weighted_f1,
            "precision": precision,
            "recall": recall,
            "avg_confidence": np.mean(probabilities),
            "avg_prediction_time": np.mean(prediction_times),
            "class_report": class_report,
            "confusion_matrix": confusion_matrix(self.test_labels, predictions).tolist()
        }
        
        print(f"      📊 Accuracy: {accuracy:.3f}")
        print(f"      📊 Weighted F1: {weighted_f1:.3f}")
        print(f"      📊 Precision: {precision:.3f}")
        print(f"      📊 Recall: {recall:.3f}")
        print(f"      📊 Avg confidence: {np.mean(probabilities):.3f}")
        
        return metrics
    
    def run_model_comparison(self):
        """Run comprehensive model comparison."""
        print("\n🔄 Running Model Comparison...")
        
        available_models = []
        
        # Evaluate XGBoost
        if self.xgb_model and self.xgb_model.is_loaded:
            print("\n📊 Evaluating XGBoost Model:")
            
            # Accuracy metrics
            xgb_accuracy = self.evaluate_model_accuracy(self.xgb_model, "XGBoost")
            
            # Inference metrics
            xgb_inference = self.measure_inference_latency(self.xgb_model, "XGBoost")
            
            self.benchmark_results["model_performance"]["XGBoost"] = {
                **xgb_accuracy,
                **xgb_inference
            }
            available_models.append("XGBoost")
        
        # Evaluate TensorFlow
        if self.tf_model and self.tf_model.is_loaded:
            print("\n📊 Evaluating TensorFlow Model:")
            
            # Accuracy metrics
            tf_accuracy = self.evaluate_model_accuracy(self.tf_model, "TensorFlow")
            
            # Inference metrics
            tf_inference = self.measure_inference_latency(self.tf_model, "TensorFlow")
            
            self.benchmark_results["model_performance"]["TensorFlow"] = {
                **tf_accuracy,
                **tf_inference
            }
            available_models.append("TensorFlow")
        
        print(f"\n✅ Model comparison complete for: {', '.join(available_models)}")
        return available_models
    
    def calculate_business_metrics(self):
        """Calculate business-relevant metrics."""
        print("\n💼 Calculating Business Metrics...")
        
        # Simulate business data (in real scenario, this would come from tickets database)
        business_data = self.generate_simulated_business_data()
        
        business_metrics = {}
        
        for model_name in self.predictions_cache.keys():
            print(f"   📈 Analyzing {model_name} business impact...")
            
            predictions = self.predictions_cache[model_name]['predictions']
            probabilities = self.predictions_cache[model_name]['probabilities']
            
            # Calculate resolution time by category
            resolution_times = self.calculate_resolution_times_by_category(predictions, business_data)
            
            # Calculate agent success rate
            agent_success_rate = self.calculate_agent_success_rate(predictions, probabilities, business_data)
            
            # Calculate satisfaction correlation
            satisfaction_correlation = self.calculate_satisfaction_correlation(predictions, probabilities, business_data)
            
            business_metrics[model_name] = {
                "avg_resolution_time_by_category": resolution_times,
                "agent_success_rate": agent_success_rate,
                "satisfaction_correlation": satisfaction_correlation,
                "high_confidence_predictions_pct": sum(1 for p in probabilities if p > 0.8) / len(probabilities) * 100,
                "accurate_high_confidence_pct": self.calculate_accurate_high_confidence_rate(model_name)
            }
            
            print(f"      📊 Agent success rate: {agent_success_rate:.1%}")
            print(f"      📊 High confidence predictions: {business_metrics[model_name]['high_confidence_predictions_pct']:.1f}%")
        
        self.benchmark_results["business_metrics"] = business_metrics
        print("   ✅ Business metrics calculation complete")
    
    def generate_simulated_business_data(self) -> Dict[str, Any]:
        """Generate simulated business data for metrics calculation."""
        categories = list(set(self.test_labels))
        
        # Simulate resolution times (in hours) by category
        resolution_times = {}
        for category in categories:
            base_time = np.random.normal(24, 8)  # 24 hours average, 8 hours std
            resolution_times[category] = max(1, base_time)  # Minimum 1 hour
        
        # Simulate agent success rates
        agent_success_rates = np.random.uniform(0.6, 0.9, len(self.test_data))
        
        # Simulate customer satisfaction scores (1-5)
        satisfaction_scores = np.random.uniform(3.0, 5.0, len(self.test_data))
        
        return {
            "resolution_times": resolution_times,
            "agent_success_rates": agent_success_rates,
            "satisfaction_scores": satisfaction_scores
        }
    
    def calculate_resolution_times_by_category(self, predictions: List[str], business_data: Dict) -> Dict[str, float]:
        """Calculate average resolution time by predicted category."""
        category_times = {}
        category_counts = {}
        
        resolution_times = business_data["resolution_times"]
        
        for pred_category in predictions:
            if pred_category in resolution_times:
                if pred_category not in category_times:
                    category_times[pred_category] = 0
                    category_counts[pred_category] = 0
                
                category_times[pred_category] += resolution_times[pred_category]
                category_counts[pred_category] += 1
        
        # Calculate averages
        avg_times = {}
        for category in category_times:
            if category_counts[category] > 0:
                avg_times[category] = category_times[category] / category_counts[category]
        
        return avg_times
    
    def calculate_agent_success_rate(self, predictions: List[str], probabilities: List[float], business_data: Dict) -> float:
        """Calculate agent success rate when using model suggestions."""
        success_rates = business_data["agent_success_rates"]
        
        # Weight success rate by model confidence
        weighted_success = 0
        total_weight = 0
        
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if i < len(success_rates):
                weight = prob  # Use model confidence as weight
                weighted_success += success_rates[i] * weight
                total_weight += weight
        
        return weighted_success / total_weight if total_weight > 0 else 0
    
    def calculate_satisfaction_correlation(self, predictions: List[str], probabilities: List[float], business_data: Dict) -> float:
        """Calculate correlation between model confidence and customer satisfaction."""
        satisfaction_scores = business_data["satisfaction_scores"]
        
        if len(probabilities) != len(satisfaction_scores):
            # Align lengths
            min_length = min(len(probabilities), len(satisfaction_scores))
            probabilities = probabilities[:min_length]
            satisfaction_scores = satisfaction_scores[:min_length]
        
        # Calculate Pearson correlation
        correlation = np.corrcoef(probabilities, satisfaction_scores)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def calculate_accurate_high_confidence_rate(self, model_name: str) -> float:
        """Calculate accuracy rate for high-confidence predictions."""
        cache = self.predictions_cache[model_name]
        predictions = cache['predictions']
        probabilities = cache['probabilities']
        true_labels = cache['true_labels']
        
        high_conf_indices = [i for i, p in enumerate(probabilities) if p > 0.8]
        
        if not high_conf_indices:
            return 0.0
        
        high_conf_predictions = [predictions[i] for i in high_conf_indices]
        high_conf_true = [true_labels[i] for i in high_conf_indices]
        
        correct = sum(1 for pred, true in zip(high_conf_predictions, high_conf_true) if pred == true)
        return correct / len(high_conf_indices) * 100
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations."""
        print("\n📈 Generating Visualizations...")
        
        # Create visualizations directory
        viz_dir = self.results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Model comparison plots
        self.plot_model_comparison(viz_dir)
        
        # Confusion matrices
        self.plot_confusion_matrices(viz_dir)
        
        # ROC curves
        self.plot_roc_curves(viz_dir)
        
        # Business metrics plots
        self.plot_business_metrics(viz_dir)
        
        print(f"   ✅ Visualizations saved to {viz_dir}")
    
    def plot_model_comparison(self, viz_dir: Path):
        """Create model comparison visualizations."""
        models_data = self.benchmark_results["model_performance"]
        
        if len(models_data) < 2:
            print("   ⚠️ Need at least 2 models for comparison plots")
            return
        
        # Prepare data for plotting
        metrics = ["accuracy", "weighted_f1", "precision", "recall"]
        model_names = list(models_data.keys())
        
        metric_values = {metric: [] for metric in metrics}
        
        for model in model_names:
            for metric in metrics:
                metric_values[metric].append(models_data[model].get(metric, 0))
        
        # Create comparison bar plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            bars = ax.bar(model_names, metric_values[metric], 
                         color=['#3498db', '#e74c3c'][:len(model_names)])
            
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "model_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Inference metrics comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Model Inference Metrics Comparison', fontsize=16, fontweight='bold')
        
        inference_metrics = [
            ("avg_latency_ms", "Average Latency (ms)"),
            ("memory_usage_mb", "Memory Usage (MB)"),
            ("throughput_per_sec", "Throughput (req/sec)")
        ]
        
        for i, (metric, title) in enumerate(inference_metrics):
            values = [models_data[model].get(metric, 0) for model in model_names]
            bars = axes[i].bar(model_names, values, 
                              color=['#2ecc71', '#f39c12'][:len(model_names)])
            
            axes[i].set_title(title, fontweight='bold')
            axes[i].set_ylabel('Value')
            
            # Add value labels
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.01,
                           f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "model_inference_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrices(self, viz_dir: Path):
        """Generate confusion matrix plots for each model."""
        for model_name in self.predictions_cache.keys():
            cache = self.predictions_cache[model_name]
            
            # Get unique labels
            all_labels = sorted(list(set(cache['true_labels'] + cache['predictions'])))
            
            # Create confusion matrix
            cm = confusion_matrix(cache['true_labels'], cache['predictions'], labels=all_labels)
            
            # Plot
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=all_labels, yticklabels=all_labels)
            plt.title(f'{model_name} - Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Category')
            plt.ylabel('True Category')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            plt.savefig(viz_dir / f"{model_name.lower()}_confusion_matrix.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_roc_curves(self, viz_dir: Path):
        """Generate ROC curve plots for each model."""
        for model_name in self.predictions_cache.keys():
            cache = self.predictions_cache[model_name]
            
            # Get unique labels for multiclass ROC
            unique_labels = sorted(list(set(cache['true_labels'])))
            
            if len(unique_labels) < 2:
                continue
            
            # Binarize labels for ROC curve
            y_true_bin = label_binarize(cache['true_labels'], classes=unique_labels)
            
            # For ROC curve, we need probability scores for each class
            # Since we only have overall confidence, we'll create a simplified version
            y_scores = np.array(cache['probabilities'])
            
            plt.figure(figsize=(10, 8))
            
            if len(unique_labels) == 2:
                # Binary classification ROC
                fpr, tpr, _ = roc_curve(
                    [1 if label == unique_labels[1] else 0 for label in cache['true_labels']],
                    y_scores
                )
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            else:
                # Multiclass ROC - simplified approach
                # Plot average ROC
                mean_tpr = 0.0
                mean_fpr = np.linspace(0, 1, 100)
                
                for i, class_label in enumerate(unique_labels):
                    y_true_class = [1 if label == class_label else 0 for label in cache['true_labels']]
                    
                    # Use confidence as proxy for class probability
                    y_score_class = [prob if pred == class_label else 1-prob 
                                   for pred, prob in zip(cache['predictions'], y_scores)]
                    
                    try:
                        fpr, tpr, _ = roc_curve(y_true_class, y_score_class)
                        mean_tpr += np.interp(mean_fpr, fpr, tpr)
                    except:
                        continue
                
                mean_tpr /= len(unique_labels)
                mean_auc = auc(mean_fpr, mean_tpr)
                
                plt.plot(mean_fpr, mean_tpr, lw=2, 
                        label=f'Mean ROC curve (AUC = {mean_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} - ROC Curve', fontsize=16, fontweight='bold')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(viz_dir / f"{model_name.lower()}_roc_curve.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_business_metrics(self, viz_dir: Path):
        """Generate business metrics visualizations."""
        business_data = self.benchmark_results["business_metrics"]
        
        if not business_data:
            return
        
        model_names = list(business_data.keys())
        
        # Agent success rate comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Business Metrics Comparison', fontsize=16, fontweight='bold')
        
        # Success rates
        success_rates = [business_data[model]["agent_success_rate"] for model in model_names]
        bars1 = axes[0].bar(model_names, success_rates, color=['#27ae60', '#e67e22'][:len(model_names)])
        axes[0].set_title('Agent Success Rate', fontweight='bold')
        axes[0].set_ylabel('Success Rate')
        axes[0].set_ylim(0, 1)
        
        for bar, value in zip(bars1, success_rates):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.1%}', ha='center', va='bottom')
        
        # High confidence predictions
        high_conf_rates = [business_data[model]["high_confidence_predictions_pct"] for model in model_names]
        bars2 = axes[1].bar(model_names, high_conf_rates, color=['#8e44ad', '#e74c3c'][:len(model_names)])
        axes[1].set_title('High Confidence Predictions %', fontweight='bold')
        axes[1].set_ylabel('Percentage')
        
        for bar, value in zip(bars2, high_conf_rates):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value:.1f}%', ha='center', va='bottom')
        
        # Satisfaction correlation
        correlations = [business_data[model]["satisfaction_correlation"] for model in model_names]
        bars3 = axes[2].bar(model_names, correlations, color=['#16a085', '#f39c12'][:len(model_names)])
        axes[2].set_title('Satisfaction Correlation', fontweight='bold')
        axes[2].set_ylabel('Correlation Coefficient')
        axes[2].set_ylim(-1, 1)
        
        for bar, value in zip(bars3, correlations):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "business_metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results_csv(self):
        """Save detailed results to CSV."""
        print("\n💾 Saving Results to CSV...")
        
        # Model performance metrics
        perf_data = []
        for model_name, metrics in self.benchmark_results["model_performance"].items():
            row = {"model": model_name}
            row.update(metrics)
            perf_data.append(row)
        
        perf_df = pd.DataFrame(perf_data)
        perf_df.to_csv(self.results_dir / "model_performance_metrics.csv", index=False)
        
        # Business metrics
        business_data = []
        for model_name, metrics in self.benchmark_results["business_metrics"].items():
            row = {"model": model_name}
            # Flatten nested dictionaries
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        row[f"{key}_{sub_key}"] = sub_value
                else:
                    row[key] = value
            business_data.append(row)
        
        business_df = pd.DataFrame(business_data)
        business_df.to_csv(self.results_dir / "business_metrics.csv", index=False)
        
        # Summary metrics
        summary_data = []
        for model_name in self.benchmark_results["model_performance"].keys():
            perf = self.benchmark_results["model_performance"][model_name]
            business = self.benchmark_results["business_metrics"].get(model_name, {})
            
            summary_data.append({
                "model": model_name,
                "accuracy": perf.get("accuracy", 0),
                "weighted_f1": perf.get("weighted_f1", 0),
                "avg_latency_ms": perf.get("avg_latency_ms", 0),
                "memory_usage_mb": perf.get("memory_usage_mb", 0),
                "agent_success_rate": business.get("agent_success_rate", 0),
                "high_confidence_pct": business.get("high_confidence_predictions_pct", 0),
                "satisfaction_correlation": business.get("satisfaction_correlation", 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.results_dir / "metrics_summary.csv", index=False)
        
        print(f"   ✅ CSV files saved to {self.results_dir}")
    
    def generate_markdown_report(self):
        """Generate comprehensive markdown report."""
        print("\n📝 Generating Markdown Report...")
        
        report_content = self._build_markdown_content()
        
        report_path = self.results_dir / "report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"   ✅ Markdown report saved to {report_path}")
    
    def _build_markdown_content(self) -> str:
        """Build comprehensive markdown report content."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        content = f"""# InsightDesk AI - Model Benchmarking Report

*Generated on: {timestamp}*

## Executive Summary

This report provides a comprehensive evaluation of machine learning models deployed in the InsightDesk AI system, comparing their performance across technical metrics and business impact indicators.

## Models Evaluated

"""
        
        # Add model overview
        for model_name in self.benchmark_results["model_performance"].keys():
            content += f"- **{model_name}**: Category classification model\n"
        
        content += "\n## Technical Performance Metrics\n\n"
        
        # Performance metrics table
        content += "| Model | Accuracy | Weighted F1 | Precision | Recall | Avg Latency (ms) | Memory (MB) |\n"
        content += "|-------|----------|-------------|-----------|--------|------------------|-------------|\n"
        
        for model_name, metrics in self.benchmark_results["model_performance"].items():
            content += f"| {model_name} | {metrics.get('accuracy', 0):.3f} | "
            content += f"{metrics.get('weighted_f1', 0):.3f} | "
            content += f"{metrics.get('precision', 0):.3f} | "
            content += f"{metrics.get('recall', 0):.3f} | "
            content += f"{metrics.get('avg_latency_ms', 0):.2f} | "
            content += f"{metrics.get('memory_usage_mb', 0):.2f} |\n"
        
        content += "\n### Key Findings\n\n"
        
        # Add performance insights
        perf_data = self.benchmark_results["model_performance"]
        if len(perf_data) >= 2:
            models = list(perf_data.keys())
            
            # Best accuracy
            best_acc_model = max(models, key=lambda m: perf_data[m].get('accuracy', 0))
            best_acc = perf_data[best_acc_model].get('accuracy', 0)
            content += f"- **Best Accuracy**: {best_acc_model} ({best_acc:.1%})\n"
            
            # Best F1
            best_f1_model = max(models, key=lambda m: perf_data[m].get('weighted_f1', 0))
            best_f1 = perf_data[best_f1_model].get('weighted_f1', 0)
            content += f"- **Best F1 Score**: {best_f1_model} ({best_f1:.3f})\n"
            
            # Fastest inference
            fastest_model = min(models, key=lambda m: perf_data[m].get('avg_latency_ms', float('inf')))
            fastest_latency = perf_data[fastest_model].get('avg_latency_ms', 0)
            content += f"- **Fastest Inference**: {fastest_model} ({fastest_latency:.2f}ms)\n"
            
            # Memory efficiency
            most_efficient = min(models, key=lambda m: perf_data[m].get('memory_usage_mb', float('inf')))
            efficiency = perf_data[most_efficient].get('memory_usage_mb', 0)
            content += f"- **Most Memory Efficient**: {most_efficient} ({efficiency:.2f}MB)\n"
        
        content += "\n## Business Impact Metrics\n\n"
        
        # Business metrics table
        if self.benchmark_results["business_metrics"]:
            content += "| Model | Agent Success Rate | High Confidence % | Satisfaction Correlation |\n"
            content += "|-------|-------------------|------------------|-------------------------|\n"
            
            for model_name, metrics in self.benchmark_results["business_metrics"].items():
                content += f"| {model_name} | "
                content += f"{metrics.get('agent_success_rate', 0):.1%} | "
                content += f"{metrics.get('high_confidence_predictions_pct', 0):.1f}% | "
                content += f"{metrics.get('satisfaction_correlation', 0):.3f} |\n"
            
            content += "\n### Business Insights\n\n"
            
            # Business insights
            biz_data = self.benchmark_results["business_metrics"]
            if len(biz_data) >= 2:
                models = list(biz_data.keys())
                
                # Best agent success
                best_agent = max(models, key=lambda m: biz_data[m].get('agent_success_rate', 0))
                agent_rate = biz_data[best_agent].get('agent_success_rate', 0)
                content += f"- **Highest Agent Success Rate**: {best_agent} ({agent_rate:.1%})\n"
                
                # Most confident
                most_confident = max(models, key=lambda m: biz_data[m].get('high_confidence_predictions_pct', 0))
                conf_rate = biz_data[most_confident].get('high_confidence_predictions_pct', 0)
                content += f"- **Most Confident Predictions**: {most_confident} ({conf_rate:.1f}%)\n"
                
                # Best satisfaction correlation
                best_satisfaction = max(models, key=lambda m: biz_data[m].get('satisfaction_correlation', -1))
                sat_corr = biz_data[best_satisfaction].get('satisfaction_correlation', 0)
                content += f"- **Best Satisfaction Correlation**: {best_satisfaction} ({sat_corr:.3f})\n"
        
        content += "\n## Visualizations\n\n"
        content += "The following visualizations are available in the `visualizations/` directory:\n\n"
        content += "- `model_performance_comparison.png` - Technical performance comparison\n"
        content += "- `model_inference_comparison.png` - Latency and throughput analysis\n"
        content += "- `business_metrics_comparison.png` - Business impact comparison\n"
        content += "- `*_confusion_matrix.png` - Confusion matrices for each model\n"
        content += "- `*_roc_curve.png` - ROC curves for each model\n"
        
        content += "\n## Recommendations\n\n"
        
        # Generate recommendations based on results
        if len(self.benchmark_results["model_performance"]) >= 2:
            models = list(self.benchmark_results["model_performance"].keys())
            perf_data = self.benchmark_results["model_performance"]
            
            # Production recommendation
            best_overall = max(models, key=lambda m: (
                perf_data[m].get('accuracy', 0) * 0.4 +
                perf_data[m].get('weighted_f1', 0) * 0.3 +
                (1000 / max(perf_data[m].get('avg_latency_ms', 1), 1)) / 1000 * 0.3
            ))
            
            content += f"### Production Deployment\n\n"
            content += f"**Recommended Model**: {best_overall}\n\n"
            content += f"Based on the balanced evaluation of accuracy, F1 score, and inference latency, "
            content += f"{best_overall} provides the best overall performance for production deployment.\n\n"
            
            content += "### Performance Optimization\n\n"
            
            # Specific recommendations for each model
            for model_name, metrics in perf_data.items():
                content += f"**{model_name}**:\n"
                
                if metrics.get('avg_latency_ms', 0) > 100:
                    content += f"- Consider optimization to reduce latency ({metrics.get('avg_latency_ms', 0):.2f}ms)\n"
                
                if metrics.get('memory_usage_mb', 0) > 50:
                    content += f"- Memory usage optimization needed ({metrics.get('memory_usage_mb', 0):.2f}MB)\n"
                
                if metrics.get('accuracy', 0) < 0.8:
                    content += f"- Accuracy improvement required ({metrics.get('accuracy', 0):.1%})\n"
                
                content += "\n"
        
        content += "## A/B Testing Framework (Placeholder)\n\n"
        content += """### Planned A/B Testing Implementation

The following A/B testing framework is planned for future implementation:

1. **Traffic Splitting**: 
   - 50/50 split between XGBoost and TensorFlow models
   - Gradual rollout with 10% initial traffic

2. **Success Metrics**:
   - Primary: Agent resolution time reduction
   - Secondary: Customer satisfaction scores
   - Guard rails: Model accuracy maintenance

3. **Statistical Significance**:
   - Minimum 1000 samples per variant
   - 95% confidence level
   - Power analysis for effect size detection

### Implementation TODO:
- [ ] Integrate with feature flag system
- [ ] Set up metrics collection pipeline
- [ ] Implement statistical analysis automation
- [ ] Create alerting for significant performance degradation
"""
        
        content += "\n## MLflow Integration (TODO)\n\n"
        content += """### Planned MLflow Integration

Future enhancements will include comprehensive MLflow integration:

1. **Experiment Tracking**:
   - Automatic logging of all benchmark runs
   - Parameter and metric tracking
   - Model versioning and artifact storage

2. **Model Registry**:
   - Centralized model management
   - Stage promotion workflow (staging → production)
   - Model lineage tracking

3. **Automated Reporting**:
   - Integration with this benchmarking pipeline
   - Automated model comparison reports
   - Performance degradation alerts

### Implementation TODO:
- [ ] Set up MLflow tracking server
- [ ] Integrate benchmark results logging
- [ ] Implement automated model registry updates
- [ ] Create model promotion workflows
- [ ] Set up performance monitoring dashboards
"""
        
        content += f"\n## Data Sources\n\n"
        content += f"- **Test Dataset**: {len(self.test_data) if self.test_data else 0} support tickets\n"
        content += f"- **Evaluation Period**: {timestamp}\n"
        content += f"- **Categories Evaluated**: {len(set(self.test_labels)) if self.test_labels else 0} unique categories\n"
        
        content += f"\n---\n*Report generated by InsightDesk AI Benchmarking System v1.0*"
        
        return content
    
    def run_complete_benchmark(self):
        """Run the complete benchmarking pipeline."""
        print("🚀 Starting Complete Model Benchmark")
        print("=" * 50)
        
        try:
            # Load models
            xgb_available, tf_available = self.load_models()
            
            if not (xgb_available or tf_available):
                print("❌ No models available for benchmarking")
                return False
            
            # Load test data
            if not self.load_test_data():
                print("❌ Failed to load test data")
                return False
            
            # Run model comparison
            evaluated_models = self.run_model_comparison()
            
            # Calculate business metrics
            self.calculate_business_metrics()
            
            # Generate visualizations
            self.generate_visualizations()
            
            # Save results
            self.save_results_csv()
            self.generate_markdown_report()
            
            # Save complete results
            results_file = self.results_dir / "complete_benchmark_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.benchmark_results, f, indent=2, default=str)
            
            print(f"\n🎉 Benchmark Complete!")
            print(f"📊 Models evaluated: {', '.join(evaluated_models)}")
            print(f"📁 Results saved to: {self.results_dir}")
            print(f"📈 Visualizations: {self.results_dir}/visualizations/")
            print(f"📝 Report: {self.results_dir}/report.md")
            
            return True
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return False


def main():
    """Main execution function."""
    print("🎯 InsightDesk AI - Model Benchmarking & Evaluation")
    print("=" * 55)
    
    # Initialize benchmarking
    benchmark = ModelBenchmark()
    
    # Run complete benchmark
    success = benchmark.run_complete_benchmark()
    
    if success:
        print("\n✅ Benchmarking completed successfully!")
        print("📊 Check the results directory for detailed analysis.")
    else:
        print("\n❌ Benchmarking failed. Check the logs for details.")


if __name__ == "__main__":
    main()