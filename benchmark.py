import os
import sys
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

def find_dataset():
    candidates = [
        "creditcard.csv",
        os.path.expanduser("~/ml-benchmark/creditcard.csv"),
        "ml-benchmark/creditcard.csv",
        os.path.join(os.path.dirname(__file__), "creditcard.csv") if "__file__" in globals() else ""
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None

def main():
    print("=" * 60)
    print("  LIGHTGBM BENCHMARK ON CREDIT CARD FRAUD DETECTION")
    print("=" * 60)

    dataset_path = find_dataset()
    if not dataset_path:
        print("[ERROR] creditcard.csv not found!")
        print("Please download dataset via Kaggle CLI first:")
        print("  kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/")
        sys.exit(1)

    print(f"[*] Loading dataset from: {dataset_path}")
    t0 = time.time()
    df = pd.read_csv(dataset_path)
    load_time_sec = time.time() - t0
    print(f"[+] Loaded {len(df):,} rows and {df.shape[1]} columns in {load_time_sec:.2f}s")

    # Features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Stratified train/test split
    print("[*] Splitting dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train / validation split for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    print("[*] Training LightGBM model...")
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    t_train_start = time.time()
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    train_time_sec = time.time() - t_train_start
    best_iteration = int(model.best_iteration_ or model.n_estimators)
    print(f"[+] Training completed in {train_time_sec:.2f}s (Best iteration: {best_iteration})")

    # Evaluate on test set
    print("[*] Evaluating model on test set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, y_pred_proba))
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))

    # Benchmark: Inference latency for 1 row (average over 100 runs)
    sample_row = X_test.iloc[0:1]
    # Warm-up
    for _ in range(10):
        _ = model.predict_proba(sample_row)

    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        _ = model.predict_proba(sample_row)
        latencies.append(time.perf_counter() - start)
    latency_1_row_ms = float(np.mean(latencies) * 1000)

    # Benchmark: Inference throughput for 1000 rows
    sample_1000 = X_test.iloc[0:1000]
    # Warm-up
    _ = model.predict_proba(sample_1000)

    t_start = time.perf_counter()
    _ = model.predict_proba(sample_1000)
    batch_time_sec = time.perf_counter() - t_start
    throughput_1000_rows_sec = float(1000 / batch_time_sec) if batch_time_sec > 0 else 0.0

    # Summary dictionary
    results = {
        "load_time_sec": round(load_time_sec, 4),
        "train_time_sec": round(train_time_sec, 4),
        "best_iteration": best_iteration,
        "auc_roc": round(auc, 4),
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "inference_latency_1_row_ms": round(latency_1_row_ms, 4),
        "inference_throughput_1000_rows_per_sec": round(throughput_1000_rows_sec, 2),
        "batch_time_1000_rows_sec": round(batch_time_sec, 6)
    }

    # Save to benchmark_result.json
    output_json_path = "benchmark_result.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"[+] Saved results to {output_json_path}")

    # Print markdown table format for report
    print("\n" + "=" * 60)
    print("  KẾT QUẢ BENCHMARK (ĐIỀN VÀO BÁO CÁO)")
    print("=" * 60)
    print(f"| Metric | Kết quả |")
    print(f"|---|---|")
    print(f"| Thời gian load data | {results['load_time_sec']:.2f} s |")
    print(f"| Thời gian training | {results['train_time_sec']:.2f} s |")
    print(f"| Best iteration | {results['best_iteration']} |")
    print(f"| AUC-ROC | {results['auc_roc']:.4f} |")
    print(f"| Accuracy | {results['accuracy']:.4f} |")
    print(f"| F1-Score | {results['f1_score']:.4f} |")
    print(f"| Precision | {results['precision']:.4f} |")
    print(f"| Recall | {results['recall']:.4f} |")
    print(f"| Inference latency (1 row) | {results['inference_latency_1_row_ms']:.2f} ms |")
    print(f"| Inference throughput (1000 rows) | {results['inference_throughput_1000_rows_per_sec']:.1f} rows/s |")
    print("=" * 60)

if __name__ == "__main__":
    main()
