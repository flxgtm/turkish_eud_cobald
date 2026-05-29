import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import TrainingArguments

from cobald_parser import CobaldParserConfig, CobaldParser
from src.processing import (
    transform_dataset,
    extract_unique_labels,
    build_schema_with_class_labels,
    replace_none_with_ignore_index,
    collate_with_padding,
    LEMMA_RULE,
    JOINT_FEATS,
    UD_DEPREL,
    EUD_DEPREL,
    MISC,
    SEMCLASS,
    DEEPSLOT,
)
from src.trainer import CustomTrainer


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def update_vocabulary(config: CobaldParserConfig, features) -> None:
    for column in [LEMMA_RULE, JOINT_FEATS, UD_DEPREL, EUD_DEPREL, MISC, DEEPSLOT, SEMCLASS]:
        if column in features:
            labels = features[column].feature.names
            config.vocabulary[column] = dict(enumerate(labels))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CoBaLDParser on the Turkish EUD JSONL dataset."
    )

    parser.add_argument(
        "--train-file",
        type=Path,
        default=REPO_ROOT / "data" / "jsonl" / "train.jsonl",
        help="Path to the training JSONL file.",
    )
    parser.add_argument(
        "--dev-file",
        type=Path,
        default=REPO_ROOT / "data" / "jsonl" / "dev.jsonl",
        help="Path to the development JSONL file.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPO_ROOT / "code" / "CobaldParser" / "model_config.json",
        help="Path to the CoBaLDParser model configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "models" / "eud_baseline",
        help="Directory where the trained model will be saved.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size for training and evaluation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(read_jsonl(args.train_file)),
            "validation": Dataset.from_list(read_jsonl(args.dev_file)),
        }
    )

    dataset_dict = transform_dataset(dataset_dict)

    print(dataset_dict)
    print("Columns:", dataset_dict["train"].column_names)

    tagsets = build_tagsets(dataset_dict)
    schema = build_schema_with_class_labels(tagsets)

    dataset_dict = (
        dataset_dict
        .cast(schema)
        .map(replace_none_with_ignore_index)
        .with_format("torch")
    )

    model_config = CobaldParserConfig.from_json_file(str(args.model_config))
    update_vocabulary(model_config, dataset_dict["train"].features)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        remove_unused_columns=False,
        report_to=[],
    )

    training_args.label_names = ["counting_masks"]
    train_features = dataset_dict["train"].features

    for dataset_column, parser_input in (
        (LEMMA_RULE, "lemma_rules"),
        (JOINT_FEATS, "joint_feats"),
        (UD_DEPREL, "deps_ud"),
        (EUD_DEPREL, "deps_eud"),
        (MISC, "miscs"),
        (DEEPSLOT, "deepslots"),
        (SEMCLASS, "semclasses"),
    ):
        if dataset_column in train_features and dataset_column in model_config.vocabulary:
            training_args.label_names.append(parser_input)

    print("Label names:", training_args.label_names)

    model = CobaldParser(model_config)

    print("Train size:", len(dataset_dict["train"]))
    print("Validation size:", len(dataset_dict["validation"]))

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        data_collator=collate_with_padding,
        compute_metrics=None,
    )

    trainer.train(ignore_keys_for_eval=["words", "sent_ids", "texts"])
    trainer.save_model()


if __name__ == "__main__":
    main()