"""FedAvg (McMahan et al., 2017) and SCAFFOLD (Karimireddy et al., 2020), simulated on one machine."""
import copy
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .metrics import evaluate


@dataclass
class FLConfig:
    algorithm: str = "fedavg"     # "fedavg" or "scaffold"
    rounds: int = 15
    local_epochs: int = 1
    client_fraction: float = 1.0
    lr: float = 1e-3              # local (client) learning rate
    global_lr: float = 1.0        # server step size (SCAFFOLD's eta_g)
    optimizer: str = "adam"       # "adam" or "sgd"; SCAFFOLD + "sgd" is the thesis' "SCAFFOLD optimizer"
    weight_decay: float = 0.0
    device: str = "cpu"


class FederatedTrainer:
    def __init__(self, model, client_loaders, cfg: FLConfig, seed=0):
        self.cfg = cfg
        self.clients = client_loaders
        self.global_model = model.to(cfg.device)
        self.scaffold = cfg.algorithm == "scaffold"
        self.rng = random.Random(seed)
        if self.scaffold:
            zeros = lambda: [torch.zeros_like(p) for p in self.global_model.parameters()]
            self.c = zeros()
            self.c_i = [zeros() for _ in client_loaders]

    def run(self, val_loader=None, log=print):
        history = []
        n = len(self.clients)
        for r in range(1, self.cfg.rounds + 1):
            selected = sorted(self.rng.sample(range(n), max(round(self.cfg.client_fraction * n), 1)))
            updates = [self._local_update(k) for k in selected]
            self._aggregate(updates)
            if val_loader is not None:
                m = evaluate(self.global_model, val_loader, self.cfg.device)
                m.pop("confusion")
                history.append({"round": r, **m})
                log(f"round {r:3d} | " + " ".join(f"{k} {v:.4f}" for k, v in m.items()))
        return history

    def _optimizer(self, model):
        opt = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[self.cfg.optimizer]
        return opt(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def _local_update(self, k):
        model = copy.deepcopy(self.global_model)
        model.train()
        opt = self._optimizer(model)
        steps = 0
        for _ in range(self.cfg.local_epochs):
            for x, y in self.clients[k]:
                x, y = x.to(self.cfg.device), y.to(self.cfg.device)
                opt.zero_grad()
                F.cross_entropy(model(x), y).backward()
                if self.scaffold:  # drift correction: g_i(y_i) + c - c_i
                    for p, c, ci in zip(model.parameters(), self.c, self.c_i[k]):
                        p.grad += c - ci
                opt.step()
                steps += 1

        update = {"state": model.state_dict(), "n": len(self.clients[k].dataset)}
        if self.scaffold:
            # Option II: c_i+ = c_i - c + (x - y_i) / (K * eta_l)
            new_ci = [
                ci - c + (x.detach() - y.detach()) / (steps * self.cfg.lr)
                for ci, c, x, y in zip(self.c_i[k], self.c, self.global_model.parameters(), model.parameters())
            ]
            update["delta_c"] = [a - b for a, b in zip(new_ci, self.c_i[k])]
            self.c_i[k] = new_ci
        return update

    @torch.no_grad()
    def _aggregate(self, updates):
        if self.scaffold:
            weights = [1 / len(updates)] * len(updates)
        else:
            total = sum(u["n"] for u in updates)
            weights = [u["n"] / total for u in updates]

        # BatchNorm running statistics are averaged together with the weights.
        state = self.global_model.state_dict()
        for key, x in state.items():
            if not x.is_floating_point():
                x.copy_(updates[0]["state"][key])
                continue
            avg = sum(w * u["state"][key] for w, u in zip(weights, updates))
            x.add_(self.cfg.global_lr * (avg - x))

        if self.scaffold:
            for j, c in enumerate(self.c):
                c.add_(sum(u["delta_c"][j] for u in updates) / len(self.clients))
