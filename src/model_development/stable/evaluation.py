import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_predictions(y_true: pd.Series, predictions_dict: dict) -> dict:
    """
    Stable evaluation logic for comparing models consistently.
    Takes a dictionary of model_name -> predicted_probabilities.
    Returns AUROC and AUPR for each model.
    """
    results = {}
    print("\n=== Evaluation Results ===")
    for name, preds in predictions_dict.items():
        auc = roc_auc_score(y_true, preds)
        ap = average_precision_score(y_true, preds)
        results[name] = {'AUROC': auc, 'AUPR': ap}
        
        print(f"{name}:")
        print(f"  AUROC: {auc:.4f}")
        print(f"  AUPR:  {ap:.4f}")
        
    return results
