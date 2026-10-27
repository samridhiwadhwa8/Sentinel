"""Train and evaluate a RoBERTa faithfulness classifier with LoRA.

CSV input must contain ``context``, ``statement``, and ``label`` columns.
Labels are ``UNFAITHFUL``/``FAITHFUL`` or 0/1 respectively. An optional
predefined split column can contain ``train``, ``validation``, and ``test``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

UNFAITHFUL = 0
FAITHFUL = 1
LABEL2ID = {"UNFAITHFUL": UNFAITHFUL, "FAITHFUL": FAITHFUL}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}


def _normalize_labels(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.upper()
    normalized = normalized.replace({"0": "UNFAITHFUL", "1": "FAITHFUL"})
    unknown = sorted(set(normalized) - set(LABEL2ID))
    if unknown:
        raise ValueError(
            "Labels must be UNFAITHFUL/FAITHFUL or 0/1; "
            f"unsupported values: {unknown}"
        )
    return normalized.map(LABEL2ID).astype(int)


def load_examples(
    csv_path: Path,
    context_column: str,
    statement_column: str,
    label_column: str,
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required = {context_column, statement_column, label_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    frame = frame.dropna(subset=[context_column, statement_column, label_column]).copy()
    frame[context_column] = frame[context_column].astype(str).str.strip()
    frame[statement_column] = frame[statement_column].astype(str).str.strip()
    frame = frame[
        frame[context_column].ne("") & frame[statement_column].ne("")
    ].copy()
    if frame.empty:
        raise ValueError("CSV contains no usable labeled context/statement examples")

    frame["label_id"] = _normalize_labels(frame[label_column])
    counts = frame["label_id"].value_counts()
    if set(counts.index) != {UNFAITHFUL, FAITHFUL}:
        raise ValueError("Training data must include both faithfulness labels")
    if counts.min() < 3:
        raise ValueError(
            "Need at least three examples of each label for train/validation/test splits"
        )
    return frame.reset_index(drop=True)


def make_splits(
    frame: pd.DataFrame,
    split_column: str | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split_column:
        if split_column not in frame.columns:
            raise ValueError(f"Split column {split_column!r} does not exist")
        split_values = frame[split_column].astype(str).str.strip().str.lower()
        if not set(split_values).issubset({"train", "validation", "test"}):
            raise ValueError("Split column values must be train, validation, or test")
        splits = tuple(
            frame.loc[split_values.eq(name)].copy()
            for name in ("train", "validation", "test")
        )
        for name, split in zip(("train", "validation", "test"), splits):
            if set(split["label_id"]) != {UNFAITHFUL, FAITHFUL}:
                raise ValueError(f"The {name} split must contain both labels")
        return splits

    warnings.warn(
        "Using a row-stratified split. Related examples from the same source may "
        "leak across splits; use a preassigned split column grouped by source/document "
        "for a stronger evaluation.",
        stacklevel=2,
    )
    train, remainder = train_test_split(
        frame,
        test_size=0.2,
        random_state=seed,
        stratify=frame["label_id"],
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.5,
        random_state=seed,
        stratify=remainder["label_id"],
    )
    return train.copy(), validation.copy(), test.copy()


def metrics_at_threshold(
    labels: np.ndarray,
    unfaithful_probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    actual = np.asarray(labels, dtype=int)
    probabilities = np.asarray(unfaithful_probabilities, dtype=float)
    predicted = np.where(
        probabilities >= threshold,
        UNFAITHFUL,
        FAITHFUL,
    )
    matrix = confusion_matrix(actual, predicted, labels=[UNFAITHFUL, FAITHFUL])
    return {
        "unfaithful_probability_threshold": float(threshold),
        "accuracy": float(accuracy_score(actual, predicted)),
        "unfaithful_recall": float(
            recall_score(actual, predicted, pos_label=UNFAITHFUL, zero_division=0)
        ),
        "unfaithful_precision": float(
            precision_score(actual, predicted, pos_label=UNFAITHFUL, zero_division=0)
        ),
        "unfaithful_f1": float(
            f1_score(actual, predicted, pos_label=UNFAITHFUL, zero_division=0)
        ),
        "confusion_matrix": {
            "labels": ["UNFAITHFUL", "FAITHFUL"],
            "rows_actual_columns_predicted": matrix.tolist(),
        },
    }


def select_unfaithful_threshold(
    labels: np.ndarray,
    unfaithful_probabilities: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    probabilities = np.asarray(unfaithful_probabilities, dtype=float)
    candidates = np.unique(np.concatenate(([0.0, 1.0], probabilities)))
    results = [
        metrics_at_threshold(labels, probabilities, float(candidate))
        for candidate in candidates
    ]
    best = max(
        results,
        key=lambda result: (
            result["unfaithful_f1"],
            result["unfaithful_recall"],
            result["accuracy"],
        ),
    )
    return best["unfaithful_probability_threshold"], best


def risk_coverage(labels: np.ndarray, faithful_probabilities: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(labels, dtype=int)
    faithful_scores = np.asarray(faithful_probabilities, dtype=float)
    order = np.argsort(-faithful_scores, kind="stable")
    sorted_labels = actual[order]
    cumulative_unfaithful = np.cumsum(sorted_labels == UNFAITHFUL)
    coverage = np.arange(1, len(actual) + 1, dtype=float) / len(actual)
    risk = cumulative_unfaithful / np.arange(1, len(actual) + 1, dtype=float)
    return {
        "aurc": float(auc(np.concatenate(([0.0], coverage)), np.concatenate(([0.0], risk)))),
        "coverage": coverage.tolist(),
        "unfaithful_risk": risk.tolist(),
    }


def evaluate_model(
    model: Any,
    tokenizer: Any,
    frame: pd.DataFrame,
    context_column: str,
    statement_column: str,
    batch_size: int,
    device: Any,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import DataCollatorWithPadding

    class EvaluationDataset(Dataset):
        def __init__(self) -> None:
            self.encodings = tokenizer(
                frame[context_column].tolist(),
                frame[statement_column].tolist(),
                truncation=True,
                max_length=512,
            )

        def __len__(self) -> int:
            return len(frame)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {key: values[index] for key, values in self.encodings.items()}

    loader = DataLoader(
        EvaluationDataset(),
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt"),
    )
    model.eval()
    logits = []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits.append(model(**batch).logits.cpu())
    probabilities = torch.softmax(torch.cat(logits), dim=-1).numpy()
    return frame["label_id"].to_numpy(), probabilities


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
    import torch.nn.functional as functional

    frame = load_examples(
        args.data,
        args.context_column,
        args.statement_column,
        args.label_column,
    )
    train_frame, validation_frame, test_frame = make_splits(
        frame, args.split_column, args.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    class TrainingDataset(Dataset):
        def __init__(self, examples: pd.DataFrame) -> None:
            self.examples = examples.reset_index(drop=True)
            self.encodings = tokenizer(
                self.examples[args.context_column].tolist(),
                self.examples[args.statement_column].tolist(),
                truncation=True,
                max_length=512,
            )

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = {key: values[index] for key, values in self.encodings.items()}
            item["labels"] = int(self.examples.iloc[index]["label_id"])
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type="single_label_classification",
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["query", "value"],
            modules_to_save=["classifier"],
        ),
    )

    counts = train_frame["label_id"].value_counts()
    class_weights = torch.tensor(
        [len(train_frame) / (2 * counts.get(UNFAITHFUL, 1)),
         len(train_frame) / (2 * counts.get(FAITHFUL, 1))],
        dtype=torch.float,
    )

    def compute_metrics(prediction: Any) -> dict[str, float]:
        scores = prediction.predictions
        if isinstance(scores, tuple):
            scores = scores[0]
        predicted = np.argmax(scores, axis=-1)
        labels = prediction.label_ids
        return {
            "accuracy": float(accuracy_score(labels, predicted)),
            "unfaithful_f1": float(
                f1_score(labels, predicted, pos_label=UNFAITHFUL, zero_division=0)
            ),
            "unfaithful_recall": float(
                recall_score(labels, predicted, pos_label=UNFAITHFUL, zero_division=0)
            ),
        }

    class WeightedTrainer(Trainer):
        def compute_loss(
            self,
            current_model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            labels = inputs.pop("labels")
            outputs = current_model(**inputs)
            loss = functional.cross_entropy(
                outputs.logits,
                labels,
                weight=class_weights.to(outputs.logits.device),
            )
            return (loss, outputs) if return_outputs else loss

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="unfaithful_f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=TrainingDataset(train_frame),
        eval_dataset=TrainingDataset(validation_frame),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    model = trainer.model
    model.eval()

    _, validation_probabilities = evaluate_model(
        model,
        tokenizer,
        validation_frame,
        args.context_column,
        args.statement_column,
        args.batch_size,
        trainer.args.device,
    )
    validation_labels = validation_frame["label_id"].to_numpy()
    selected_unfaithful_threshold, validation_threshold_metrics = (
        select_unfaithful_threshold(
            validation_labels, validation_probabilities[:, UNFAITHFUL]
        )
    )

    test_labels, test_probabilities = evaluate_model(
        model,
        tokenizer,
        test_frame,
        args.context_column,
        args.statement_column,
        args.batch_size,
        trainer.args.device,
    )
    test_metrics = metrics_at_threshold(
        test_labels, test_probabilities[:, UNFAITHFUL], selected_unfaithful_threshold
    )
    test_curve = risk_coverage(test_labels, test_probabilities[:, FAITHFUL])
    comparison = None

    if args.baseline_model:
        baseline_path = Path(args.baseline_model)
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline model not found: {baseline_path}")
        baseline_tokenizer = AutoTokenizer.from_pretrained(str(baseline_path))
        baseline_model = AutoModelForSequenceClassification.from_pretrained(
            str(baseline_path)
        ).to(trainer.args.device)
        _, baseline_validation_probabilities = evaluate_model(
            baseline_model,
            baseline_tokenizer,
            validation_frame,
            args.context_column,
            args.statement_column,
            args.batch_size,
            trainer.args.device,
        )
        baseline_threshold, baseline_validation_metrics = select_unfaithful_threshold(
            validation_frame["label_id"].to_numpy(),
            baseline_validation_probabilities[:, UNFAITHFUL],
        )
        baseline_test_labels, baseline_test_probabilities = evaluate_model(
            baseline_model,
            baseline_tokenizer,
            test_frame,
            args.context_column,
            args.statement_column,
            args.batch_size,
            trainer.args.device,
        )
        comparison = {
            "model": str(baseline_path),
            "validation_at_selected_threshold": baseline_validation_metrics,
            "test_at_validation_threshold": metrics_at_threshold(
                baseline_test_labels,
                baseline_test_probabilities[:, UNFAITHFUL],
                baseline_threshold,
            ),
            "test_risk_coverage": risk_coverage(
                baseline_test_labels,
                baseline_test_probabilities[:, FAITHFUL],
            ),
        }

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metadata = {
        "base_model": args.base_model,
        "adapter": "LoRA",
        "label2id": LABEL2ID,
        "threshold_selection": "validation maximum UNFAITHFUL F1",
        "unfaithful_probability_threshold": selected_unfaithful_threshold,
        "faithfulness_accept_threshold": 1.0 - selected_unfaithful_threshold,
        "validation_at_selected_threshold": validation_threshold_metrics,
        "test_at_validation_threshold": test_metrics,
        "test_risk_coverage": test_curve,
        "baseline_comparison": comparison,
        "split_sizes": {
            "train": len(train_frame),
            "validation": len(validation_frame),
            "test": len(test_frame),
        },
        "split_column": args.split_column,
        "warning": (
            None if args.split_column else
            "Random row split used; group/source leakage may inflate test metrics."
        ),
    }
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (args.output_dir / "threshold.json").write_text(
        json.dumps(
            {"faithfulness_accept_threshold": 1.0 - selected_unfaithful_threshold},
            indent=2,
        ),
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Labeled examples CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/sentinel_faithfulness_roberta_lora"),
    )
    parser.add_argument("--base-model", default="roberta-base")
    parser.add_argument(
        "--baseline-model",
        help="Optional trained DistilBERT checkpoint to compare on the same splits",
    )
    parser.add_argument("--context-column", default="context")
    parser.add_argument("--statement-column", default="statement")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--split-column",
        help="Optional preassigned train/validation/test column; prefer source-grouped splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
