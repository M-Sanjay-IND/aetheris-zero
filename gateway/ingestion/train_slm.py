import argparse
import json
import os
import sys
from pathlib import Path
import time
from typing import Dict, List, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from gateway.ingestion.neural_slm_model import (
    BRICK_CLASSES,
    EQUIPMENT_TYPES,
    POINT_ROLES,
    SUBSYSTEMS,
    UNITS,
    ZONE_CLASSES,
    AetherisBrickSLM,
    BMSTokenizer,
)


class BMSTagDataset(Dataset):
    def __init__(self, records: List[dict], tokenizer: BMSTokenizer):
        self.records = records
        self.tokenizer = tokenizer

        self.class_to_idx = {c: i for i, c in enumerate(BRICK_CLASSES)}
        self.eq_to_idx = {e: i for i, e in enumerate(EQUIPMENT_TYPES)}
        self.role_to_idx = {r: i for i, r in enumerate(POINT_ROLES)}
        self.sub_to_idx = {s: i for i, s in enumerate(SUBSYSTEMS)}
        self.zone_to_idx = {z: i for i, z in enumerate(ZONE_CLASSES)}
        self.unit_to_idx = {u: i for i, u in enumerate(UNITS)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.records[idx]
        tokens = self.tokenizer.encode(row["raw_tag"])

        c_idx = self.class_to_idx.get(row.get("brick_class", ""), 0)
        eq_idx = self.eq_to_idx.get(row.get("equipment_type", ""), 0)
        role_idx = self.role_to_idx.get(row.get("point_role", ""), 0)
        sub_idx = self.sub_to_idx.get(row.get("subsystem", ""), 0)
        z_val = row.get("zone_id") or "unassigned"
        z_idx = self.zone_to_idx.get(z_val, self.zone_to_idx["unassigned"])
        u_idx = self.unit_to_idx.get(row.get("unit", "unknown"), self.unit_to_idx["unknown"])

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "brick_class": torch.tensor(c_idx, dtype=torch.long),
            "equipment_type": torch.tensor(eq_idx, dtype=torch.long),
            "point_role": torch.tensor(role_idx, dtype=torch.long),
            "subsystem": torch.tensor(sub_idx, dtype=torch.long),
            "zone_id": torch.tensor(z_idx, dtype=torch.long),
            "unit": torch.tensor(u_idx, dtype=torch.long),
        }


from core.device_utils import get_optimal_device


def train_neural_slm(
    data_path: Union[str, Path],
    save_checkpoint_path: Union[str, Path] = "models/checkpoints/slm_brick_best.pt",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 2e-3,
    force_cpu: bool = False,
) -> Tuple[AetherisBrickSLM, Dict[str, float]]:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = get_optimal_device(force_cpu)
    print(f"[Aetheris Brick-SLM] Initializing training on device: {device}")

    # Load dataset
    data_file = Path(data_path)
    if not data_file.exists():
        from data.datasets.dataset_generator import build_and_save_all_datasets
        build_and_save_all_datasets()

    records = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    print(f"[Aetheris Brick-SLM] Loaded {len(records)} training samples from {data_file}")

    # Train / Val Split (85% / 15%)
    rng = np.random.default_rng(42)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    split_idx = int(0.85 * len(records))

    train_records = [records[i] for i in indices[:split_idx]]
    val_records = [records[i] for i in indices[split_idx:]]

    model = AetherisBrickSLM().to(device)
    train_dataset = BMSTagDataset(train_records, model.tokenizer)
    val_dataset = BMSTagDataset(val_records, model.tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_metrics = {}

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_acc = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != "input_ids"}

            optimizer.zero_grad()
            logits = model(input_ids)
            total_loss, _ = model.compute_loss(logits, targets)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_acc += total_loss.item()

        scheduler.step()
        train_loss_avg = train_loss_acc / len(train_loader)

        # Validation evaluation
        model.eval()
        correct_class = 0
        correct_eq = 0
        correct_role = 0
        total_val = 0
        val_loss_acc = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                targets = {k: v.to(device) for k, v in batch.items() if k != "input_ids"}
                logits = model(input_ids)
                v_loss, _ = model.compute_loss(logits, targets)
                val_loss_acc += v_loss.item()

                pred_c = torch.argmax(logits["brick_class"], dim=-1)
                pred_eq = torch.argmax(logits["equipment_type"], dim=-1)
                pred_role = torch.argmax(logits["point_role"], dim=-1)

                correct_class += (pred_c == targets["brick_class"]).sum().item()
                correct_eq += (pred_eq == targets["equipment_type"]).sum().item()
                correct_role += (pred_role == targets["point_role"]).sum().item()
                total_val += len(input_ids)

        val_loss_avg = val_loss_acc / len(val_loader)
        acc_class = correct_class / total_val
        acc_eq = correct_eq / total_val
        acc_role = correct_role / total_val
        composite_acc = (acc_class + acc_eq + acc_role) / 3.0

        if composite_acc > best_val_acc:
            best_val_acc = composite_acc
            model.save_checkpoint(save_checkpoint_path)
            best_metrics = {
                "epoch": epoch,
                "val_loss": round(val_loss_avg, 4),
                "accuracy_class": round(acc_class, 4),
                "accuracy_equipment": round(acc_eq, 4),
                "accuracy_role": round(acc_role, 4),
                "composite_accuracy": round(composite_acc, 4),
            }

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss_avg:.4f} | "
                f"Val Loss: {val_loss_avg:.4f} | Class Acc: {acc_class*100:.1f}% | "
                f"Eq Acc: {acc_eq*100:.1f}% | Role Acc: {acc_role*100:.1f}%"
            )

    duration = round(time.time() - start_time, 2)
    print(f"[Aetheris Brick-SLM] Completed in {duration}s. Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"[Aetheris Brick-SLM] Checkpoint saved to: {save_checkpoint_path}")

    return model, best_metrics


def export_generative_slm_instruction_dataset(
    jsonl_input: Union[str, Path] = "data/datasets/slm_bacnet_brick_corpus.jsonl",
    output_path: Union[str, Path] = "data/datasets/slm_instruction_tuning.json",
):
    """Export formatted instruction-tuning JSON for generative SLMs (Phi-3.5, Qwen2.5, Llama-3.2)."""
    input_file = Path(jsonl_input)
    if not input_file.exists():
        from data.datasets.dataset_generator import build_and_save_all_datasets
        build_and_save_all_datasets()

    dataset = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            instruction = (
                "You are an expert building ontology and Brick Schema v1.3 normalization engine. "
                "Analyze the provided raw building automation point tag string and extract the standardized "
                "Brick class, equipment type, point role, subsystem, zone assignment, and engineering unit."
            )
            input_text = f"Point Tag: {item['raw_tag']}"
            output_data = {
                "brick_class": item["brick_class"],
                "equipment_type": item["equipment_type"],
                "equipment_id": item["equipment_id"],
                "point_role": item["point_role"],
                "subsystem": item["subsystem"],
                "zone_id": item["zone_id"],
                "unit": item["unit"],
                "param_key": item.get("param_key", "none"),
            }
            dataset.append({
                "instruction": instruction,
                "input": input_text,
                "output": json.dumps(output_data, indent=2)
            })

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"[Instruction Fine-Tuning] Exported {len(dataset)} instruction pairs to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Aetheris Brick-SLM Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--export-instructions", action="store_true", help="Export generative instruction JSON")
    args = parser.parse_args()

    data_path = Path("data/datasets/slm_bacnet_brick_corpus.jsonl")
    checkpoint_path = Path("models/checkpoints/slm_brick_best.pt")

    train_neural_slm(
        data_path=data_path,
        save_checkpoint_path=checkpoint_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        force_cpu=args.cpu
    )

    if args.export_instructions:
        export_generative_slm_instruction_dataset()
