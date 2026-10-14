"""Offline benchmark for prioritizing missed dealership calls for human callback."""

import json
import math
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

DATA_PATH = Path(__file__).with_name("callback_triage_cases.json")
FEATURE_FIELDS = (
    "manager",
    "undrivable",
    "safety",
    "repeat",
    "appointment",
    "routine",
    "sentiment",
    "delay_over_day",
)


def _classification_metrics(expected: list[int], predicted: list[int]) -> dict:
    true_positives = sum(actual == 1 and guess == 1 for actual, guess in zip(expected, predicted))
    true_negatives = sum(actual == 0 and guess == 0 for actual, guess in zip(expected, predicted))
    false_positives = sum(actual == 0 and guess == 1 for actual, guess in zip(expected, predicted))
    false_negatives = sum(actual == 1 and guess == 0 for actual, guess in zip(expected, predicted))
    return {
        "precision": round(precision_score(expected, predicted, zero_division=0), 4),
        "recall": round(recall_score(expected, predicted, zero_division=0), 4),
        "f1": round(f1_score(expected, predicted, zero_division=0), 4),
        "confusion_matrix_tn_fp_fn_tp": [
            [true_negatives, false_positives],
            [false_negatives, true_positives],
        ],
    }


def _rule_score(example: dict) -> float:
    return (
        4 * example["safety"]
        + 3 * example["undrivable"]
        + 3 * example["manager"]
        + example["repeat"]
        + example["delay_over_day"]
    )


def _capacity_metrics(expected: list[int], scores: list[float], capacity: int) -> dict:
    ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    selected = ranked[:capacity]
    urgent_found = sum(expected[index] == 1 for index in selected)
    total_urgent = sum(expected)
    return {
        "capacity": capacity,
        "precision_at_capacity": round(urgent_found / capacity, 4) if capacity else None,
        "recall_at_capacity": round(urgent_found / total_urgent, 4) if total_urgent else None,
        "urgent_calls_found": urgent_found,
    }


def run_benchmark() -> dict:
    dataset = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    examples = dataset["cases"]
    train = [example for example in examples if example["split"] == "train"]
    test = [example for example in examples if example["split"] == "test"]
    if not train or not test or {example["urgent_callback"] for example in train} != {0, 1}:
        raise ValueError("Dataset requires train and test rows, with both labels in train.")

    vectorizer = DictVectorizer(sparse=False)
    train_features = vectorizer.fit_transform([
        {key: row[key] for key in FEATURE_FIELDS} for row in train
    ])
    test_features = vectorizer.transform([
        {key: row[key] for key in FEATURE_FIELDS} for row in test
    ])
    labels_train = [row["urgent_callback"] for row in train]
    labels_test = [row["urgent_callback"] for row in test]

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=7)
    model.fit(train_features, labels_train)
    model_scores = model.predict_proba(test_features)[:, 1].tolist()
    model_predictions = [int(score >= 0.5) for score in model_scores]

    rules_scores = [_rule_score(row) for row in test]
    rules_predictions = [int(score >= 3) for score in rules_scores]
    callback_capacity = max(1, math.ceil(len(test) * 0.25))

    def summarize(predictions: list[int], scores: list[float]) -> dict:
        return {
            **_classification_metrics(labels_test, predictions),
            **_capacity_metrics(labels_test, scores, callback_capacity),
        }

    return {
        "task": dataset["task"],
        "label_definition": dataset["label_definition"],
        "data_note": "Synthetic teaching data only. Metrics do not estimate dealership performance.",
        "train_rows": len(train),
        "test_rows": len(test),
        "positive_rate_test": round(sum(labels_test) / len(labels_test), 4),
        "callback_capacity_assumption": f"top {callback_capacity} of {len(test)} test cases (25%)",
        "rule_baseline": summarize(rules_predictions, rules_scores),
        "logistic_regression": summarize(model_predictions, model_scores),
        "test_predictions": [
            {
                "id": row["id"],
                "label": row["urgent_callback"],
                "rule_score": rules_scores[index],
                "rule_prediction": rules_predictions[index],
                "model_probability": round(model_scores[index], 4),
                "model_prediction": model_predictions[index],
            }
            for index, row in enumerate(test)
        ],
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))
