# scripts/demo_benchmarking.py

"""
Benchmarking Demo Script

This script demonstrates the benchmarking and evaluation capabilities
using sample data and mock results when models aren't available.
"""

import sys
import json
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.benchmark_models import ModelBenchmark


def create_sample_results():
    """Create comprehensive sample benchmark results."""
    return {
        "model_performance": {
            "XGBoost": {
                "accuracy": 0.834,
                "weighted_f1": 0.829,
                "precision": 0.841,
                "recall": 0.834,
                "avg_confidence": 0.756,
                "avg_prediction_time": 0.0452,
                "avg_latency_ms": 45.2,
                "p95_latency_ms": 78.1,
                "p99_latency_ms": 120.5,
                "memory_usage_mb": 23.4,
                "success_rate": 1.0,
                "throughput_per_sec": 22.1,
                "class_report": {
                    "authentication": {"precision": 0.85, "recall": 0.82, "f1-score": 0.84, "support": 450},
                    "payment": {"precision": 0.83, "recall": 0.86, "f1-score": 0.84, "support": 380},
                    "performance": {"precision": 0.84, "recall": 0.83, "f1-score": 0.84, "support": 320},
                    "infrastructure": {"precision": 0.82, "recall": 0.81, "f1-score": 0.82, "support": 280},
                    "ui": {"precision": 0.86, "recall": 0.85, "f1-score": 0.85, "support": 270}
                },
                "confusion_matrix": [
                    [369, 23, 15, 32, 11],
                    [18, 327, 12, 15, 8],
                    [21, 18, 266, 10, 5],
                    [28, 22, 8, 227, 15],
                    [15, 12, 8, 6, 229]
                ]
            },
            "TensorFlow": {
                "accuracy": 0.847,
                "weighted_f1": 0.842,
                "precision": 0.853,
                "recall": 0.847,
                "avg_confidence": 0.781,
                "avg_prediction_time": 0.0678,
                "avg_latency_ms": 67.8,
                "p95_latency_ms": 112.3,
                "p99_latency_ms": 165.2,
                "memory_usage_mb": 45.7,
                "success_rate": 1.0,
                "throughput_per_sec": 14.7,
                "class_report": {
                    "authentication": {"precision": 0.87, "recall": 0.85, "f1-score": 0.86, "support": 450},
                    "payment": {"precision": 0.85, "recall": 0.88, "f1-score": 0.87, "support": 380},
                    "performance": {"precision": 0.86, "recall": 0.85, "f1-score": 0.85, "support": 320},
                    "infrastructure": {"precision": 0.84, "recall": 0.83, "f1-score": 0.84, "support": 280},
                    "ui": {"precision": 0.88, "recall": 0.87, "f1-score": 0.87, "support": 270}
                },
                "confusion_matrix": [
                    [383, 20, 12, 25, 10],
                    [15, 334, 10, 12, 9],
                    [18, 15, 272, 8, 7],
                    [25, 18, 6, 232, 9],
                    [12, 8, 5, 10, 235]
                ]
            }
        },
        "business_metrics": {
            "XGBoost": {
                "agent_success_rate": 0.782,
                "high_confidence_predictions_pct": 68.3,
                "satisfaction_correlation": 0.342,
                "accurate_high_confidence_pct": 89.1,
                "avg_resolution_time_by_category": {
                    "authentication": 18.5,
                    "payment": 24.2,
                    "performance": 32.7,
                    "infrastructure": 28.4,
                    "ui": 15.3
                }
            },
            "TensorFlow": {
                "agent_success_rate": 0.809,
                "high_confidence_predictions_pct": 73.7,
                "satisfaction_correlation": 0.387,
                "accurate_high_confidence_pct": 91.4,
                "avg_resolution_time_by_category": {
                    "authentication": 16.8,
                    "payment": 22.1,
                    "performance": 29.3,
                    "infrastructure": 26.7,
                    "ui": 14.2
                }
            }
        },
        "timestamp": "2025-09-30T01:00:00"
    }


def create_sample_predictions_cache():
    """Create sample predictions cache for visualization."""
    import random
    
    categories = ["authentication", "payment", "performance", "infrastructure", "ui"]
    
    xgb_predictions = []
    xgb_probabilities = []
    tf_predictions = []
    tf_probabilities = []
    true_labels = []
    
    # Generate 300 samples
    for i in range(300):
        true_category = random.choice(categories)
        true_labels.append(true_category)
        
        # XGBoost predictions (slightly lower accuracy)
        if random.random() < 0.83:  # 83% accuracy
            xgb_pred = true_category
            xgb_conf = random.uniform(0.7, 0.95)
        else:
            xgb_pred = random.choice([c for c in categories if c != true_category])
            xgb_conf = random.uniform(0.5, 0.8)
        
        xgb_predictions.append(xgb_pred)
        xgb_probabilities.append(xgb_conf)
        
        # TensorFlow predictions (slightly higher accuracy)
        if random.random() < 0.85:  # 85% accuracy
            tf_pred = true_category
            tf_conf = random.uniform(0.75, 0.95)
        else:
            tf_pred = random.choice([c for c in categories if c != true_category])
            tf_conf = random.uniform(0.5, 0.8)
        
        tf_predictions.append(tf_pred)
        tf_probabilities.append(tf_conf)
    
    return {
        "XGBoost": {
            "predictions": xgb_predictions,
            "probabilities": xgb_probabilities,
            "true_labels": true_labels
        },
        "TensorFlow": {
            "predictions": tf_predictions,
            "probabilities": tf_probabilities,
            "true_labels": true_labels
        }
    }


def demonstrate_benchmarking():
    """Demonstrate the complete benchmarking pipeline."""
    print("🎯 InsightDesk AI - Benchmarking Demonstration")
    print("=" * 50)
    
    print("""
This demo showcases comprehensive model benchmarking capabilities:

🔹 Model Performance Evaluation
  - Accuracy, Precision, Recall, F1-Score
  - Inference latency and memory usage
  - Confusion matrices and ROC curves
  
🔹 Business Impact Analysis  
  - Agent success rates
  - Resolution time improvements
  - Customer satisfaction correlation
  
🔹 Comprehensive Reporting
  - CSV exports for analysis
  - Markdown reports with insights
  - Visual comparisons and charts
""")
    
    # Initialize benchmark system
    print("\n📊 Initializing Benchmarking System...")
    benchmark = ModelBenchmark(results_dir="demo_results")
    
    # Load sample results
    print("📁 Loading sample benchmark results...")
    benchmark.benchmark_results = create_sample_results()
    benchmark.predictions_cache = create_sample_predictions_cache()
    
    print("✅ Sample data loaded successfully")
    
    # Generate comprehensive analysis
    print("\n🔄 Generating Comprehensive Analysis...")
    
    print("   📈 Creating visualizations...")
    benchmark.generate_visualizations()
    print("   ✅ Visualizations complete")
    
    print("   💾 Exporting to CSV...")
    benchmark.save_results_csv()
    print("   ✅ CSV export complete")
    
    print("   📝 Generating markdown report...")
    benchmark.generate_markdown_report()
    print("   ✅ Report generation complete")
    
    # Display key insights
    print("\n📊 Key Performance Insights")
    print("=" * 30)
    
    perf_data = benchmark.benchmark_results["model_performance"]
    business_data = benchmark.benchmark_results["business_metrics"]
    
    print("\n🎯 Model Comparison Summary:")
    for model_name in perf_data.keys():
        perf = perf_data[model_name]
        business = business_data[model_name]
        
        print(f"\n📋 {model_name}:")
        print(f"   Accuracy: {perf['accuracy']:.1%}")
        print(f"   F1-Score: {perf['weighted_f1']:.3f}")
        print(f"   Avg Latency: {perf['avg_latency_ms']:.1f}ms")
        print(f"   Memory Usage: {perf['memory_usage_mb']:.1f}MB")
        print(f"   Agent Success Rate: {business['agent_success_rate']:.1%}")
        print(f"   High Confidence Predictions: {business['high_confidence_predictions_pct']:.1f}%")
    
    # Recommendations
    print("\n💡 Recommendations")
    print("=" * 20)
    
    xgb_perf = perf_data["XGBoost"]
    tf_perf = perf_data["TensorFlow"]
    
    if tf_perf["accuracy"] > xgb_perf["accuracy"]:
        accuracy_winner = "TensorFlow"
        accuracy_advantage = (tf_perf["accuracy"] - xgb_perf["accuracy"]) * 100
    else:
        accuracy_winner = "XGBoost"
        accuracy_advantage = (xgb_perf["accuracy"] - tf_perf["accuracy"]) * 100
    
    if xgb_perf["avg_latency_ms"] < tf_perf["avg_latency_ms"]:
        speed_winner = "XGBoost"
        speed_advantage = tf_perf["avg_latency_ms"] - xgb_perf["avg_latency_ms"]
    else:
        speed_winner = "TensorFlow"
        speed_advantage = xgb_perf["avg_latency_ms"] - tf_perf["avg_latency_ms"]
    
    print(f"🎯 **Accuracy Leader**: {accuracy_winner} (+{accuracy_advantage:.1f}%)")
    print(f"⚡ **Speed Leader**: {speed_winner} ({speed_advantage:.1f}ms faster)")
    print(f"💼 **Business Impact**: TensorFlow shows higher agent success rate")
    print(f"📈 **Recommendation**: Consider A/B testing TensorFlow vs current XGBoost")
    
    # Show file outputs
    results_dir = Path("demo_results")
    print(f"\n📁 Generated Files")
    print("=" * 18)
    
    output_files = [
        "metrics_summary.csv",
        "model_performance_metrics.csv", 
        "business_metrics.csv",
        "report.md",
        "complete_benchmark_results.json"
    ]
    
    viz_files = [
        "model_performance_comparison.png",
        "model_inference_comparison.png",
        "business_metrics_comparison.png",
        "xgboost_confusion_matrix.png",
        "tensorflow_confusion_matrix.png",
        "xgboost_roc_curve.png",
        "tensorflow_roc_curve.png"
    ]
    
    print("📊 **Data Files**:")
    for file in output_files:
        file_path = results_dir / file
        if file_path.exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ⚠️ {file}")
    
    print("\n📈 **Visualizations**:")
    viz_dir = results_dir / "visualizations"
    for file in viz_files:
        file_path = viz_dir / file
        if file_path.exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ⚠️ {file}")
    
    # Show sample of the markdown report
    report_path = results_dir / "report.md"
    if report_path.exists():
        print(f"\n📝 Sample Report Content")
        print("=" * 25)
        
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Show first few lines
        lines = content.split('\n')
        for line in lines[:15]:
            if line.strip():
                print(f"   {line}")
        
        print("   ...")
        print(f"   📄 Full report available at: {report_path}")
    
    print(f"\n🎉 Benchmarking Demonstration Complete!")
    print(f"📁 All results saved to: {results_dir}")
    print(f"🔍 Review the generated files for detailed analysis")
    
    return True


def main():
    """Main demonstration function."""
    try:
        success = demonstrate_benchmarking()
        
        if success:
            print("\n✅ Demo completed successfully!")
            print("\n🚀 Next Steps:")
            print("   1. Review generated visualizations")
            print("   2. Analyze CSV data for trends")
            print("   3. Set up A/B testing with winning model")
            print("   4. Integrate with MLflow for tracking")
        else:
            print("\n❌ Demo failed!")
            
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")


if __name__ == "__main__":
    main()