import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import torch
from datasets import Dataset, DatasetDict
from safetensors.torch import load_file

from cobald_parser import CobaldParserConfig, CobaldParser
from src.processing import (
    transform_dataset,
    extract_unique_labels,
    build_schema_with_class_labels,
    replace_none_with_ignore_index,
    collate_with_padding,
    EUD_ARC_FROM,
    EUD_ARC_TO,
    EUD_DEPREL,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CoBaLDParser EUD predictions on dev or test split."
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to the trained model directory containing config and model.safetensors.",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="test",
        help="Evaluation split.",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=REPO_ROOT / "data" / "jsonl" / "train.jsonl",
        help="Path to the training JSONL file. Used for rebuilding the label schema.",
    )

    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    items = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    return items


def build_tagsets(dataset_dict: DatasetDict) -> dict:
    tagsets = {}

    all_column_names = set()
    for split in dataset_dict:
        all_column_names.update(dataset_dict[split].column_names)

    for column_name in all_column_names:
        labels = set()

        for split in dataset_dict:
            if column_name in dataset_dict[split].column_names:
                labels |= extract_unique_labels(dataset_dict[split], column_name)

        if labels:
            tagsets[column_name] = labels

    return tagsets


def decode_label(vocab: dict, idx: int) -> str:
    if idx in vocab:
        return vocab[idx]
    if str(idx) in vocab:
        return vocab[str(idx)]
    return f"UNK_{idx}"


def gold_set(item, vocab: dict) -> set[tuple[int, int, str]]:
    result = set()

    for arc_from, arc_to, rel_id in zip(
        item[EUD_ARC_FROM],
        item[EUD_ARC_TO],
        item[EUD_DEPREL],
    ):
        rel = decode_label(vocab, int(rel_id))
        result.add((int(arc_from), int(arc_to), rel))

    return result


def pred_set(output, vocab: dict) -> set[tuple[int, int, str]]:
    result = set()
    pred = output.get("deps_eud")

    if pred is None:
        return result

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().tolist()

    for arc in pred:
        if len(arc) != 4:
            continue

        batch_idx, arc_from, arc_to, rel_id = arc

        if int(batch_idx) != 0:
            continue

        rel = decode_label(vocab, int(rel_id))
        result.add((int(arc_from), int(arc_to), rel))

    return result


def load_dataset(train_file: Path, eval_file: Path, split_name: str) -> DatasetDict:
    raw = DatasetDict(
        {
            "train": Dataset.from_list(read_jsonl(train_file)),
            split_name: Dataset.from_list(read_jsonl(eval_file)),
        }
    )

    dataset = transform_dataset(raw)
    schema = build_schema_with_class_labels(build_tagsets(dataset))

    dataset = (
        dataset
        .cast(schema)
        .map(replace_none_with_ignore_index)
        .with_format("torch")
    )

    return dataset


def load_model(model_dir: Path) -> CobaldParser:
    config = CobaldParserConfig.from_pretrained(str(model_dir))
    model = CobaldParser(config)

    state_dict_path = model_dir / "model.safetensors"
    state_dict = load_file(str(state_dict_path))

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def evaluate(dataset: DatasetDict, model: CobaldParser, split_name: str) -> None:
    gold_vocab_eud = dict(enumerate(dataset[split_name].features[EUD_DEPREL].feature.names))
    pred_vocab_eud = model.config.vocabulary["eud_deprel"]

    tp = 0
    pred_total = 0
    gold_total = 0

    by_rel = defaultdict(lambda: Counter({"tp": 0, "pred": 0, "gold": 0}))

    examples_with_gold = 0
    examples_with_pred = 0

    with torch.no_grad():
        for item in dataset[split_name]:
            batch = collate_with_padding([item])
            output = model(**batch)

            gold = gold_set(item, gold_vocab_eud)
            pred = pred_set(output, pred_vocab_eud)

            if gold:
                examples_with_gold += 1
            if pred:
                examples_with_pred += 1

            intersection = gold & pred

            tp += len(intersection)
            pred_total += len(pred)
            gold_total += len(gold)

            for _, _, rel in intersection:
                by_rel[rel]["tp"] += 1

            for _, _, rel in pred:
                by_rel[rel]["pred"] += 1

            for _, _, rel in gold:
                by_rel[rel]["gold"] += 1

    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    jaccard = tp / (pred_total + gold_total - tp) if (pred_total + gold_total - tp) else 0.0

    print("=" * 80)
    print(f"EUD EXACT EDGE METRICS ON {split_name.upper()}")
    print("=" * 80)
    print("Gold edges:", gold_total)
    print("Predicted edges:", pred_total)
    print("True positives:", tp)
    print("Examples with gold EUD:", examples_with_gold)
    print("Examples with predicted EUD:", examples_with_pred)
    print()
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Jaccard:   {jaccard:.4f}")

    print()
    print("=" * 80)
    print("BY RELATION")
    print("=" * 80)

    for rel, counts in sorted(by_rel.items()):
        rel_tp = counts["tp"]
        rel_pred = counts["pred"]
        rel_gold = counts["gold"]

        rel_precision = rel_tp / rel_pred if rel_pred else 0.0
        rel_recall = rel_tp / rel_gold if rel_gold else 0.0
        rel_f1 = (
            2 * rel_precision * rel_recall / (rel_precision + rel_recall)
            if rel_precision + rel_recall
            else 0.0
        )

        print(
            f"{rel:12s} "
            f"gold={rel_gold:4d} "
            f"pred={rel_pred:4d} "
            f"tp={rel_tp:4d} "
            f"P={rel_precision:.4f} "
            f"R={rel_recall:.4f} "
            f"F1={rel_f1:.4f}"
        )


def main() -> None:
    args = parse_args()

    eval_file = REPO_ROOT / "data" / "jsonl" / f"{args.split}.jsonl"

    dataset = load_dataset(
        train_file=args.train_file,
        eval_file=eval_file,
        split_name=args.split,
    )
    model = load_model(args.model_dir)

    evaluate(dataset, model, args.split)


if __name__ == "__main__":
    main()