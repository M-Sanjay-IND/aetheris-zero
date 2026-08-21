import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BRICK_CLASSES = [
    "Supply_Air_Temperature_Sensor",
    "Supply_Air_Temperature_Setpoint",
    "Discharge_Air_Temperature_Setpoint",
    "Outside_Air_Temperature_Sensor",
    "Zone_Air_Temperature_Sensor",
    "Zone_Air_Temperature_Setpoint",
    "Electric_Power_Sensor",
    "Air_Flow_Sensor",
    "Damper_Position_Command",
    "Chilled_Water_Supply_Temperature_Sensor",
    "Chilled_Water_Supply_Temperature_Setpoint",
    "Chilled_Water_Return_Temperature_Sensor",
    "Solar_Radiance_Sensor",
    "Occupancy_Sensor",
    "Thermal_Capacitance_Air_Parameter",
    "Thermal_Capacitance_Mass_Parameter",
    "Envelope_Thermal_Resistance_Parameter",
    "Mass_Thermal_Resistance_Parameter",
    "Interzone_Thermal_Resistance_Parameter",
    "Floor_Area_Parameter",
    "Solar_Factor_Parameter",
    "CO2_Level_Sensor",
    "Relative_Humidity_Sensor",
    "Fan_Status",
    "Filter_Differential_Pressure_Sensor",
]

EQUIPMENT_TYPES = [
    "Air_Handling_Unit",
    "Chiller",
    "Variable_Air_Volume_Box",
    "Building",
    "Fan_Coil_Unit",
    "Boiler",
    "Cooling_Tower",
    "Pump",
    "Battery_Energy_Storage_System",
    "Electric_Vehicle_Supply_Equipment",
]

POINT_ROLES = ["sensor", "setpoint", "command", "parameter", "meter", "status"]
SUBSYSTEMS = ["hvac", "thermal_model", "electrical", "environment"]
ZONE_CLASSES = ["zone_1", "zone_2", "zone_3", "zone_4", "zone_5", "unassigned"]
UNITS = ["deg_C", "kW", "m3/s", "ratio", "count", "K/kW", "kJ/K", "m2", "W/m2", "ppm", "%", "kPa", "unknown"]


class BMSTokenizer:
    """Subword & character n-gram tokenizer designed for cryptic BMS / BACnet tag strings."""
    def __init__(self, max_vocab: int = 1000, max_seq_len: int = 48):
        self.max_seq_len = max_seq_len
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3}
        self.inv_vocab = {0: "<PAD>", 1: "<UNK>", 2: "<CLS>", 3: "<SEP>"}
        self._build_base_vocab()

    def _build_base_vocab(self):
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-/ "
        for c in chars:
            if c not in self.vocab:
                idx = len(self.vocab)
                self.vocab[c] = idx
                self.inv_vocab[idx] = c

        common_subwords = [
            "AHU", "VAV", "CHLR", "CHILLER", "BLDG", "BUILDING", "SAT", "RAT", "DAT", "OAT",
            "TEMP", "SP", "STPT", "SETPT", "PWR", "KW", "FLOW", "DMPR", "DAMPER", "POS",
            "CMD", "STAT", "STATUS", "CHW", "SUP", "RET", "SOLAR", "IRRAD", "OCC", "CO2",
            "RH", "CZ", "CM", "REXT", "RM", "RADJ", "AREA", "SQM", "FRAC", "PARAM",
            "ZONE", "ZN", "Z01", "Z02", "Z03", "Z04", "Z05", "CORE", "NORTH", "SOUTH", "EAST", "WEST",
            "FLOOR", "FL01", "FL02", "FL03", "FAN", "PUMP", "VALVE", "VLV", "MTR", "METER"
        ]
        for sw in common_subwords:
            if sw not in self.vocab:
                idx = len(self.vocab)
                self.vocab[sw] = idx
                self.inv_vocab[idx] = sw

    def encode(self, text: str) -> List[int]:
        tokens = [self.vocab["<CLS>"]]
        text_clean = text.strip()

        # Tokenize by delimiters while recognizing common subwords
        parts = []
        cur = ""
        for char in text_clean:
            if char in "_-./ ":
                if cur:
                    parts.append(cur)
                    cur = ""
                parts.append(char)
            else:
                cur += char
        if cur:
            parts.append(cur)

        for p in parts:
            p_up = p.upper()
            if p_up in self.vocab:
                tokens.append(self.vocab[p_up])
            else:
                for c in p:
                    tokens.append(self.vocab.get(c, self.vocab["<UNK>"]))

        tokens.append(self.vocab["<SEP>"])

        # Truncate or pad to max_seq_len
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len - 1] + [self.vocab["<SEP>"]]
        else:
            tokens = tokens + [self.vocab["<PAD>"]] * (self.max_seq_len - len(tokens))

        return tokens

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        return torch.tensor([self.encode(t) for t in texts], dtype=torch.long)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class AetherisBrickSLM(nn.Module):
    """
    Multi-Task Neural Small Language Model (SLM) for sub-millisecond point tag semantic parsing
    and Brick Schema v1.3 entity normalization.
    """
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        d_ff: int = 256,
        max_seq_len: int = 48,
        dropout: float = 0.1
    ):
        super().__init__()
        self.tokenizer = BMSTokenizer(max_vocab=vocab_size, max_seq_len=max_seq_len)
        self.embedding = nn.Embedding(len(self.tokenizer.vocab), d_model, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            MultiHeadAttentionBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )

        # Multi-task classification heads
        self.head_brick_class = nn.Linear(d_model, len(BRICK_CLASSES))
        self.head_equipment = nn.Linear(d_model, len(EQUIPMENT_TYPES))
        self.head_role = nn.Linear(d_model, len(POINT_ROLES))
        self.head_subsystem = nn.Linear(d_model, len(SUBSYSTEMS))
        self.head_zone = nn.Linear(d_model, len(ZONE_CLASSES))
        self.head_unit = nn.Linear(d_model, len(UNITS))

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        padding_mask = (input_ids == 0)

        x = self.embedding(input_ids) + self.pos_embedding[:, :seq_len, :]
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, key_padding_mask=padding_mask)

        # Global average pooling over non-padded tokens
        mask = (~padding_mask).unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6)
        rep = self.pooler(pooled)

        return {
            "brick_class": self.head_brick_class(rep),
            "equipment_type": self.head_equipment(rep),
            "point_role": self.head_role(rep),
            "subsystem": self.head_subsystem(rep),
            "zone_id": self.head_zone(rep),
            "unit": self.head_unit(rep),
        }

    def compute_loss(
        self,
        logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        default_weights = {
            "brick_class": 1.5,
            "equipment_type": 1.0,
            "point_role": 1.0,
            "subsystem": 0.8,
            "zone_id": 1.0,
            "unit": 0.8,
        }
        w = weights or default_weights
        losses = {}
        total_loss = torch.tensor(0.0, device=logits["brick_class"].device)

        for k in logits:
            if k in targets:
                l = F.cross_entropy(logits[k], targets[k])
                losses[k] = l.item()
                total_loss = total_loss + w.get(k, 1.0) * l

        return total_loss, losses

    @torch.no_grad()
    def predict_tags(self, tags: List[str], device: str = "cpu") -> List[Dict[str, Any]]:
        self.eval()
        self.to(device)
        input_ids = self.tokenizer.encode_batch(tags).to(device)
        logits = self.forward(input_ids)

        probs_class = F.softmax(logits["brick_class"], dim=-1).cpu().numpy()
        probs_eq = F.softmax(logits["equipment_type"], dim=-1).cpu().numpy()
        probs_role = F.softmax(logits["point_role"], dim=-1).cpu().numpy()
        probs_sub = F.softmax(logits["subsystem"], dim=-1).cpu().numpy()
        probs_zone = F.softmax(logits["zone_id"], dim=-1).cpu().numpy()
        probs_unit = F.softmax(logits["unit"], dim=-1).cpu().numpy()

        results = []
        for i, tag in enumerate(tags):
            c_idx = int(np.argmax(probs_class[i]))
            eq_idx = int(np.argmax(probs_eq[i]))
            r_idx = int(np.argmax(probs_role[i]))
            s_idx = int(np.argmax(probs_sub[i]))
            z_idx = int(np.argmax(probs_zone[i]))
            u_idx = int(np.argmax(probs_unit[i]))

            conf = float(probs_class[i][c_idx] * 0.5 + probs_eq[i][eq_idx] * 0.3 + probs_role[i][r_idx] * 0.2)
            z_name = ZONE_CLASSES[z_idx]

            results.append({
                "raw_tag": tag,
                "brick_class": BRICK_CLASSES[c_idx],
                "equipment_type": EQUIPMENT_TYPES[eq_idx],
                "point_role": POINT_ROLES[r_idx],
                "subsystem": SUBSYSTEMS[s_idx],
                "zone_id": None if z_name == "unassigned" else z_name,
                "unit": UNITS[u_idx],
                "confidence": round(conf, 4)
            })

        return results

    def save_checkpoint(self, save_path: Union[str, Path]):
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "vocab_size": len(self.tokenizer.vocab),
            "brick_classes": BRICK_CLASSES,
            "equipment_types": EQUIPMENT_TYPES,
            "point_roles": POINT_ROLES,
            "subsystems": SUBSYSTEMS,
            "zone_classes": ZONE_CLASSES,
            "units": UNITS,
        }, str(path))

    @classmethod
    def load_checkpoint(cls, checkpoint_path: Union[str, Path], device: str = "cpu") -> "AetherisBrickSLM":
        data = torch.load(str(checkpoint_path), map_location=device)
        model = cls()
        model.load_state_dict(data["state_dict"])
        model.to(device)
        model.eval()
        return model
