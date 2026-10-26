"""Trajectory- and composition-aware dataset splits (no frame-level leakage)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence, Tuple

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class FrameRecord:
    """One labelled frame with grouping keys for leakage-safe splits."""

    frame_id: str
    trajectory_id: str
    block_id: int
    composition_key: str
    # Optional content hash for manifest integrity (e.g. hash of positions/forces blob).
    payload_hash: str = ""

    def grouping_key_trajectory_block(self) -> Tuple[str, int]:
        return (self.trajectory_id, self.block_id)


@dataclass
class SplitManifest:
    """Deterministic split assignment with hashes for reproducibility."""

    seed: int
    strategy: str
    train: List[str] = field(default_factory=list)
    val: List[str] = field(default_factory=list)
    test: List[str] = field(default_factory=list)
    group_hashes: Dict[str, str] = field(default_factory=dict)
    frame_hashes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _stable_hash(parts: Sequence[str]) -> str:
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _partition_groups(
    groups: List[str],
    seed: int,
    train_frac: float,
    val_frac: float,
) -> Dict[SplitName, List[str]]:
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1")
    import random

    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def split_by_trajectory_blocks(
    frames: Sequence[FrameRecord],
    *,
    seed: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[Dict[SplitName, List[FrameRecord]], SplitManifest]:
    """
    Assign whole (trajectory_id, block_id) groups to train/val/test.

    Frames from the same trajectory time block never appear in more than one split.
    """
    group_to_frames: Dict[Tuple[str, int], List[FrameRecord]] = {}
    for fr in frames:
        key = fr.grouping_key_trajectory_block()
        group_to_frames.setdefault(key, []).append(fr)

    group_labels = [_stable_hash([k[0], str(k[1])]) for k in group_to_frames]
    label_to_key = dict(zip(group_labels, group_to_frames.keys()))
    parts = _partition_groups(group_labels, seed, train_frac, val_frac)

    out: Dict[SplitName, List[FrameRecord]] = {"train": [], "val": [], "test": []}
    manifest = SplitManifest(seed=seed, strategy="trajectory_block")
    for split_name, labels in parts.items():
        for lab in labels:
            manifest.group_hashes[lab] = lab
            for fr in group_to_frames[label_to_key[lab]]:
                out[split_name].append(fr)
                h = fr.payload_hash or _stable_hash([fr.frame_id, fr.composition_key])
                manifest.frame_hashes[fr.frame_id] = h
        setattr(manifest, split_name, [fr.frame_id for fr in out[split_name]])

    return out, manifest


def split_by_composition(
    frames: Sequence[FrameRecord],
    *,
    seed: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[Dict[SplitName, List[FrameRecord]], SplitManifest]:
    """Hold out entire composition keys (all trajectories/blocks for that composition)."""
    comp_to_frames: Dict[str, List[FrameRecord]] = {}
    for fr in frames:
        comp_to_frames.setdefault(fr.composition_key, []).append(fr)

    comp_keys = sorted(comp_to_frames)
    comp_labels = [_stable_hash([c]) for c in comp_keys]
    label_to_comp = dict(zip(comp_labels, comp_keys))
    parts = _partition_groups(comp_labels, seed, train_frac, val_frac)

    out: Dict[SplitName, List[FrameRecord]] = {"train": [], "val": [], "test": []}
    manifest = SplitManifest(seed=seed, strategy="composition")
    for split_name, labels in parts.items():
        for lab in labels:
            comp = label_to_comp[lab]
            manifest.group_hashes[lab] = _stable_hash([comp])
            for fr in comp_to_frames[comp]:
                out[split_name].append(fr)
                h = fr.payload_hash or _stable_hash([fr.frame_id, fr.composition_key])
                manifest.frame_hashes[fr.frame_id] = h
        setattr(manifest, split_name, [fr.frame_id for fr in out[split_name]])

    return out, manifest


def write_manifest(manifest: SplitManifest, path: Path | str) -> None:
    """Write manifest JSON (UTF-8, sorted keys)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
