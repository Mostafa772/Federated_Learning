"""Federated training on one institution (FNUSA), out-of-institution testing on the other (Mayo).

Example (FedAvg, weak non-IID, 40 clients):
    python scripts/train.py --algorithm fedavg --partition weak-noniid --rounds 15 --lr 1e-3 \
        --out runs/fedavg_weak
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fl_eeg import CLASS_NAMES, CNNGRU, PARTITIONS, FederatedTrainer, FLConfig, IEEGDataset, evaluate  # noqa: E402
from fl_eeg.data import balanced_indices  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = ap.add_argument_group("data")
    d.add_argument("--train-root", default="data/DatasetFnusa")
    d.add_argument("--train-index", default="data/fnusa.csv")
    d.add_argument("--test-root", default="data/DatasetMayo")
    d.add_argument("--test-index", default="data/mayo.csv")
    d.add_argument("--val-per-class", type=int, default=2000,
                   help="balanced subset of the test institution used for per-round validation")
    d.add_argument("--partition", choices=PARTITIONS, default="iid")
    d.add_argument("--alpha", type=float, default=100.0, help="Dirichlet concentration for weak-noniid")
    d.add_argument("--clients", type=int, default=40)
    d.add_argument("--batch-size", type=int, default=32)
    d.add_argument("--workers", type=int, default=2)

    t = ap.add_argument_group("training")
    t.add_argument("--algorithm", choices=["fedavg", "scaffold"], default="fedavg")
    t.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    t.add_argument("--rounds", type=int, default=15)
    t.add_argument("--local-epochs", type=int, default=1)
    t.add_argument("--client-fraction", type=float, default=1.0)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--global-lr", type=float, default=1.0)
    t.add_argument("--weight-decay", type=float, default=0.0)
    t.add_argument("--dropout", type=float, default=0.5)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", type=Path, default=Path("runs/latest"))
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers)

    train_set = IEEGDataset(args.train_root, args.train_index)
    test_set = IEEGDataset(args.test_root, args.test_index)
    parts = PARTITIONS[args.partition](train_set.targets, args.clients, rng,
                                       **({"alpha": args.alpha} if args.partition == "weak-noniid" else {}))
    clients = [loader(Subset(train_set, p), True) for p in parts if len(p)]
    val_loader = loader(Subset(test_set, balanced_indices(test_set.targets, args.val_per_class, rng)), False)
    print(f"{len(train_set)} training segments over {len(clients)} clients, {len(test_set)} test segments")

    cfg = FLConfig(algorithm=args.algorithm, rounds=args.rounds, local_epochs=args.local_epochs,
                   client_fraction=args.client_fraction, lr=args.lr, global_lr=args.global_lr,
                   optimizer=args.optimizer, weight_decay=args.weight_decay, device=device)
    trainer = FederatedTrainer(CNNGRU(dropout=args.dropout), clients, cfg, seed=args.seed)
    history = trainer.run(val_loader)

    test = evaluate(trainer.global_model, loader(test_set, False), device)
    print("test:", {k: v for k, v in test.items() if k != "confusion"})
    print("confusion (rows = true " + " / ".join(CLASS_NAMES) + "):")
    print(np.array(test["confusion"]))

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.global_model.state_dict(), args.out / "model.pt")
    with open(args.out / "results.json", "w") as fh:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "config": asdict(cfg),
                   "history": history, "test": test}, fh, indent=2)


if __name__ == "__main__":
    main()
