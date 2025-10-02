from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from typing import Dict, Any, List, Optional
from transformers import Trainer
from datasets import Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLossOptimized(torch.nn.Module):
    """Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations

    Modified for multi-class classification support.
    Original from https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
    
    Use reduced gamma_neg and gamma_pos for multi-class as class 
    imbalance is typically less severe than in multi-label problems.
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum(dim=1).mean()


class FocalLoss(torch.nn.Module):
    """Focal Loss implementation for classification tasks."""
    
    def __init__(self, alpha=1.0, gamma=2.0, num_classes=3):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CustomTrainer(Trainer):
    """Custom Trainer that supports different loss functions."""
    
    def __init__(self, loss_fn: Optional[torch.nn.Module] = None, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """Override compute_loss to use custom loss function if provided."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.loss_fn is not None:
            # Use custom loss function
            if isinstance(self.loss_fn, AsymmetricLossOptimized):
                # AsymmetricLoss expects one-hot encoded labels for multi-label classification
                # For multi-class classification, we need to convert to one-hot
                num_classes = logits.size(-1)
                labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
                loss = self.loss_fn(logits, labels_one_hot)
            else:
                # Standard loss functions expect class indices
                loss = self.loss_fn(logits, labels)
        else:
            # Use default cross-entropy loss
            loss = F.cross_entropy(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


def create_loss_function(loss_type: str, **kwargs) -> Optional[torch.nn.Module]:
    """Factory function to create loss functions based on type.
    
    Args:
        loss_type: Type of loss function ("standard", "focal", "weighted_asymmetric_focal")
        **kwargs: Additional parameters for loss functions
        
    Returns:
        Loss function instance or None for standard cross-entropy
    """
    if loss_type == "standard":
        return None  # Use default cross-entropy
    elif loss_type == "focal":
        return FocalLoss(
            alpha=kwargs.get("focal_alpha", 1.0),
            gamma=kwargs.get("focal_gamma", 2.0),
            num_classes=kwargs.get("num_classes", 3)
        )
    elif loss_type == "weighted_asymmetric_focal":
        return AsymmetricLossOptimized(
            gamma_neg=kwargs.get("gamma_neg", 2.0),  # Reduced from 4.0 for multi-class
            gamma_pos=kwargs.get("gamma_pos", 0.0),  # Reduced from 1.0 for multi-class
            clip=kwargs.get("clip", 0.05),
            eps=kwargs.get("eps", 1e-8),
            disable_torch_grad_focal_loss=kwargs.get("disable_torch_grad_focal_loss", False),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_metrics(eval_pred) -> Dict[str, Any]:
    logits, labels = eval_pred

    # Handle unexpected tuple output
    if isinstance(logits, tuple):
        logits = logits[0]

    if not isinstance(logits, np.ndarray):
        logits = np.asarray(logits)
        labels = np.asarray(labels)

    # Check shape consistency
    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits shape (batch_size, num_labels), got {logits.shape}"
        )
    if labels.ndim != 1:
        raise ValueError(f"Expected labels shape (batch_size,), got {labels.shape}")

    # Compute predictions
    predictions = np.argmax(logits, axis=-1)

    # Metrics
    total_accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_micro = f1_score(labels, predictions, average="micro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    precision_macro = precision_score(
        labels, predictions, average="macro", zero_division=0
    )
    recall_macro = recall_score(labels, predictions, average="macro", zero_division=0)
    precision_micro = precision_score(
        labels, predictions, average="micro", zero_division=0
    )
    recall_micro = recall_score(labels, predictions, average="micro", zero_division=0)

    # Per-class accuracy and F1 scores
    per_label_accuracy = {}
    per_label_f1 = {}
    unique_classes = np.unique(labels)
    for class_id in unique_classes:
        class_indices = np.where(labels == class_id)[0]
        if len(class_indices) > 0:
            class_preds = predictions[class_indices]
            class_labels = labels[class_indices]
            per_label_accuracy[f"accuracy_class_{class_id}"] = accuracy_score(
                class_labels, class_preds
            )
            per_label_f1[f"f1_class_{class_id}"] = f1_score(
                class_labels, class_preds, average="macro", zero_division=0
            )

    return {
        "accuracy": round(float(total_accuracy) * 100, 2),
        "f1_macro": round(float(f1_macro) * 100, 2),
        "f1_micro": round(float(f1_micro) * 100, 2),
        "f1_weighted": round(float(f1_weighted) * 100, 2),
        "precision_macro": round(float(precision_macro) * 100, 2),
        "recall_macro": round(float(recall_macro) * 100, 2),
        "precision_micro": round(float(precision_micro) * 100, 2),
        "recall_micro": round(float(recall_micro) * 100, 2),
        **{k: round(float(v) * 100, 2) for k, v in per_label_accuracy.items()},
        **{k: round(float(v) * 100, 2) for k, v in per_label_f1.items()},
    }


def batched_predict_metrics_trainer(
    trainer: Trainer, dataset: Dataset, batch_size: int = 64
) -> Dict[str, Any]:
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_ids: List[Any] = []

    if not dataset:
        return {}

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        chunk = dataset.select(range(start, end))
        ids = chunk["id"]
        output = trainer.predict(chunk)

        logits = output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]

        logits = np.asarray(logits)
        if logits.ndim != 2:
            raise ValueError(
                f"Expected logits shape (batch_size, num_labels), got {logits.shape}"
            )

        labels = np.asarray(output.label_ids)
        all_logits.append(logits)
        all_labels.append(labels)
        all_ids.extend(ids)

    # Concatenate all results
    logits = np.concatenate(all_logits, axis=0)
    predictions = np.argmax(logits, axis=1).tolist()
    labels = np.concatenate(all_labels, axis=0).tolist()
    ids = list(all_ids)

    id_to_logits_labels = {}
    for idx, pred, label in zip(ids, predictions, labels):
        if idx not in id_to_logits_labels.keys():
            id_to_logits_labels[idx] = [(pred, label)]
        else:
            id_to_logits_labels[idx].append((pred, label))

    ids = list(id_to_logits_labels.keys())
    values = [id_to_logits_labels[idx] for idx in ids]
    values = [max(set(tuples), key=tuples.count) for tuples in values]
    predictions = [int(tuples[0]) for tuples in values]
    labels = [int(tuples[1]) for tuples in values]

    metrics = compute_metrics((np.array(predictions), np.array(labels)))

    metrics["preds"] = predictions
    metrics["labels"] = labels
    metrics["ids"] = ids

    return metrics
