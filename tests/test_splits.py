from aimd_mlff_methods.data.splits import FrameRecord, split_by_composition, split_by_trajectory_blocks


def _frames():
    out = []
    for traj in ("t1", "t2", "t3"):
        for block in range(2):
            for i in range(3):
                out.append(
                    FrameRecord(
                        frame_id=f"{traj}-b{block}-f{i}",
                        trajectory_id=traj,
                        block_id=block,
                        composition_key="A" if traj != "t3" else "B",
                    )
                )
    return out


def test_trajectory_block_no_leakage():
    frames = _frames()
    splits, manifest = split_by_trajectory_blocks(frames, seed=42)
    train_ids = {f.frame_id for f in splits["train"]}
    test_ids = {f.frame_id for f in splits["test"]}
    assert train_ids.isdisjoint(test_ids)
    for fr in frames:
        group_mates = {
            x.frame_id
            for x in frames
            if x.trajectory_id == fr.trajectory_id and x.block_id == fr.block_id
        }
        in_train = group_mates & train_ids
        in_test = group_mates & test_ids
        assert not (in_train and in_test)
    assert manifest.seed == 42
    assert manifest.strategy == "trajectory_block"


def test_composition_holdout():
    frames = _frames()
    splits, manifest = split_by_composition(frames, seed=7, train_frac=0.5, val_frac=0.0)
    train_comps = {f.composition_key for f in splits["train"]}
    test_comps = {f.composition_key for f in splits["test"]}
    assert train_comps.isdisjoint(test_comps)
    assert manifest.strategy == "composition"


def test_split_reproducible():
    frames = _frames()
    _, m1 = split_by_trajectory_blocks(frames, seed=123)
    _, m2 = split_by_trajectory_blocks(frames, seed=123)
    assert m1.train == m2.train and m1.test == m2.test
