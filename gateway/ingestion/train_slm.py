import argparse
import json
import os
import sys
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
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
from core.device_utils import get_optimal_device


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


def evaluate_slm_split(model: AetherisBrickSLM, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_samples = 0
    correct_class = 0
    correct_eq = 0
    correct_role = 0
    correct_sub = 0
    correct_zone = 0
    correct_unit = 0
    all_correct_exact = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            bs = input_ids.size(0)
            total_samples += bs

            preds = model(input_ids)
            p_class = preds["brick_class"].argmax(dim=-1).cpu()
            p_eq = preds["equipment_type"].argmax(dim=-1).cpu()
            p_role = preds["point_role"].argmax(dim=-1).cpu()
            p_sub = preds["subsystem"].argmax(dim=-1).cpu()
            p_zone = preds["zone_id"].argmax(dim=-1).cpu()
            p_unit = preds["unit"].argmax(dim=-1).cpu()

            t_class = batch["brick_class"]
            t_eq = batch["equipment_type"]
            t_role = batch["point_role"]
            t_sub = batch["subsystem"]
            t_zone = batch["zone_id"]
            t_unit = batch["unit"]

            m_class = (p_class == t_class)
            m_eq = (p_eq == t_eq)
            m_role = (p_role == t_role)
            m_sub = (p_sub == t_sub)
            m_zone = (p_zone == t_zone)
            m_unit = (p_unit == t_unit)

            correct_class += m_class.sum().item()
            correct_eq += m_eq.sum().item()
            correct_role += m_role.sum().item()
            correct_sub += m_sub.sum().item()
            correct_zone += m_zone.sum().item()
            correct_unit += m_unit.sum().item()

            exact_match = m_class & m_eq & m_role & m_sub & m_zone & m_unit
            all_correct_exact += exact_match.sum().item()

    return {
        "exact_match_acc": all_correct_exact / max(1, total_samples),
        "class_acc": correct_class / max(1, total_samples),
        "eq_acc": correct_eq / max(1, total_samples),
        "role_acc": correct_role / max(1, total_samples),
        "sub_acc": correct_sub / max(1, total_samples),
        "zone_acc": correct_zone / max(1, total_samples),
        "unit_acc": correct_unit / max(1, total_samples),
        "total_samples": total_samples,
    }


def train_neural_slm(
    data_path: Union[str, Path] = "data/datasets/slm_train.jsonl",
    val_data_path: Union[str, Path] = "data/datasets/slm_val.jsonl",
    test_ood_path: Union[str, Path] = "data/datasets/slm_test_ood.jsonl",
    save_checkpoint_path: Union[str, Path] = "models/checkpoints/slm_brick_best.pt",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 2e-3,
    force_cpu: bool = False,
) -> Tuple[AetherisBrickSLM, Dict[str, Any]]:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = get_optimal_device(force_cpu)
    print(f"[Aetheris Brick-SLM] Initializing leak-free training on device: {device}")

    # Ensure datasets exist
    train_file = Path(data_path)
    val_file = Path(val_data_path)
    test_file = Path(test_ood_path)

    if not train_file.exists() or not val_file.exists() or not test_file.exists():
        from data.datasets.dataset_generator import build_and_save_all_datasets
        build_and_save_all_datasets()

    def _load_jsonl(p: Path) -> List[dict]:
        items = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line.strip()))
        return items

    train_records = _load_jsonl(train_file)
    val_records = _load_jsonl(val_file)
    test_ood_records = _load_jsonl(test_file)

    print(f"[Aetheris Brick-SLM] Dataset Splits (Strict Disjoint Facilities):")
    print(f" -> Train Samples (In-Distribution Facilities): {len(train_records)}")
    print(f" -> Val Samples (Disjoint Facilities):           {len(val_records)}")
    print(f" -> Test OOD Samples (Unseen Facilities/Vendors):{len(test_ood_records)}")

    tokenizer = BMSTokenizer()
    train_ds = BMSTagDataset(train_records, tokenizer)
    val_ds = BMSTagDataset(val_records, tokenizer)
    test_ood_ds = BMSTagDataset(test_ood_records, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_ood_loader = DataLoader(test_ood_ds, batch_size=batch_size, shuffle=False)

    model = AetherisBrickSLM(
        vocab_size=len(tokenizer.vocab),
        d_model=128,
        n_layers=3,
        n_heads=4,
        d_ff=256,
        max_seq_len=48,
        dropout=0.15,
    ).to(device)

    # Multi-task criterion with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    start_time = time.time()
    best_val_acc = 0.0
    history = []

    print(f"\n[Aetheris Brick-SLM] Commencing training across {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            preds = model(input_ids)

            l_class = criterion(preds["brick_class"], batch["brick_class"].to(device))
            l_eq = criterion(preds["equipment_type"], batch["equipment_type"].to(device))
            l_role = criterion(preds["point_role"], batch["point_role"].to(device))
            l_sub = criterion(preds["subsystem"], batch["subsystem"].to(device))
            l_zone = criterion(preds["zone_id"], batch["zone_id"].to(device))
            l_unit = criterion(preds["unit"], batch["unit"].to(device))

            loss = 1.8 * l_class + 1.2 * l_eq + 1.0 * l_role + 0.8 * l_sub + 1.0 * l_zone + 0.8 * l_unit
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(1, batches)

        # Evaluate on Val and OOD Test
        val_metrics = evaluate_slm_split(model, val_loader, device)
        ood_metrics = evaluate_slm_split(model, test_ood_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_exact_acc": val_metrics["exact_match_acc"],
            "val_class_acc": val_metrics["class_acc"],
            "val_role_acc": val_metrics["role_acc"],
            "ood_exact_acc": ood_metrics["exact_match_acc"],
            "ood_class_acc": ood_metrics["class_acc"],
            "ood_role_acc": ood_metrics["role_acc"],
        })

        if epoch % 2 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Exact: {val_metrics['exact_match_acc']*100:.1f}% (Class: {val_metrics['class_acc']*100:.1f}%) | "
                f"OOD Test Exact: {ood_metrics['exact_match_acc']*100:.1f}% (Class: {ood_metrics['class_acc']*100:.1f}%)"
            )

        if val_metrics["exact_match_acc"] > best_val_acc:
            best_val_acc = val_metrics["exact_match_acc"]
            ckpt_path = Path(save_checkpoint_path)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "vocab": tokenizer.vocab,
                "val_metrics": val_metrics,
                "ood_metrics": ood_metrics,
                "epoch": epoch,
            }, str(ckpt_path))

    duration = time.time() - start_time
    final_val_metrics = evaluate_slm_split(model, val_loader, device)
    final_ood_metrics = evaluate_slm_split(model, test_ood_loader, device)

    audit_summary = {
        "model_name": "AetherisBrickSLM (Multi-Task Transformer)",
        "training_duration_seconds": duration,
        "epochs": epochs,
        "parameters_count": sum(p.numel() for p in model.parameters()),
        "in_distribution_val": final_val_metrics,
        "out_of_distribution_test_zero_shot": final_ood_metrics,
        "leak_free_guarantee": "Strict Disjoint Facilities & Domains across Train, Val, and OOD Test",
        "history": history,
    }

    # Save metrics report JSON
    audit_json_path = Path("models/checkpoints/slm_audit_metrics.json")
    audit_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"\n[Aetheris Brick-SLM] Training complete in {duration:.2f}s!")
    print(f" -> Best Disjoint Val Exact Match:     {best_val_acc*100:.2f}%")
    print(f" -> Final Out-of-Distribution (OOD):  {final_ood_metrics['exact_match_acc']*100:.2f}% Exact Match | {final_ood_metrics['class_acc']*100:.2f}% Brick Class Acc")
    print(f" -> Checkpoint saved to:               {save_checkpoint_path}")
    print(f" -> Audit metrics saved to:            {audit_json_path}")

    return model, audit_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    train_neural_slm(epochs=args.epochs, batch_size=args.batch_size, force_cpu=args.cpu)
