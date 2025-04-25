from collections import defaultdict
import numpy as np

from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score



def calculate_metrics(y_true, y_pred):
    metrics = {
        "f1": f1_score(y_true, y_pred, average='macro'),
        # "roc_auc": roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr'),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "accuracy": np.mean(y_true == y_pred),
    }
    if y_pred.ndim == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')
    
    return metrics



def calculate_ste_metrics_with_bootstrap(y_true, y_pred, n_iterations=1000):
    # Bootstrap the metrics to get standard deviation (error)
    metrics = defaultdict(list)
    for _ in range(n_iterations):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
        sample_metrics = calculate_metrics(y_true_sample, y_pred_sample)
        for key, value in sample_metrics.items():
            metrics[key].append(value)
    return {key: np.std(value) for key, value in metrics.items()}

