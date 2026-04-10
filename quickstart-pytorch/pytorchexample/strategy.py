"""Custom Flower strategy with support for rotating malicious clients + FoolsGold."""

import io
import random
import time
from collections import OrderedDict
from logging import INFO
from typing import Callable

import numpy as np
import torch
from flwr.common import ArrayRecord, ConfigRecord, MetricRecord, log
from flwr.server import Grid
from flwr.serverapp.strategy import FedAvg as FlowerFedAvg
from flwr.serverapp.strategy.result import Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info


class FedAvg(FlowerFedAvg):
    """FedAvg with custom start() supporting rotating malicious clients."""

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
        num_malicious_nodes: int = 0,
        active_malicious_nodes_per_round: int = 0,
        rotate_malicious_nodes: bool = False,
    ) -> Result:
        """Execute federated learning with optional rotating malicious clients."""

        malicious_pool = list(range(num_malicious_nodes))

        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()

        # Initial server-side evaluation
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        # Expose global arrays so FoolsGold can compute deltas
        self._current_global_arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # Select active attackers for this round
            if active_malicious_nodes_per_round <= 0 or num_malicious_nodes <= 0:
                active_attackers = []
            elif rotate_malicious_nodes:
                k = min(active_malicious_nodes_per_round, len(malicious_pool))
                active_attackers = random.sample(malicious_pool, k=k)
            else:
                active_attackers = malicious_pool[
                    : min(active_malicious_nodes_per_round, len(malicious_pool))
                ]

            train_config["active-attackers"] = str(active_attackers)

            # Store global arrays BEFORE aggregation so FoolsGold can compute deltas
            self._current_global_arrays = arrays

            # ---------------- TRAIN ----------------
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays

            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # ---------------- EVALUATE (CLIENT SIDE) ----------------
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            # ---------------- EVALUATE (SERVER SIDE) ----------------
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")

        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))

        log(INFO, "")
        return result


# ---------------------------------------------------------------------------
# FoolsGold (Fung et al., 2020 — https://arxiv.org/abs/1808.04866)
# ---------------------------------------------------------------------------

class FoolsGold(FedAvg):
    """Down-weights clients whose cumulative gradient histories are
    suspiciously similar (i.e. colluding Sybils)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history = {}   # node_id -> cumulative gradient delta (flat np array)
        self.kappa = 1

    # -- helpers --

    @staticmethod
    def _flatten(array_record: ArrayRecord) -> np.ndarray:
        return np.concatenate([a.flatten() for a in array_record.to_numpy_ndarrays()])

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        dot = float(np.dot(a, b))
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        return dot / norm if norm > 1e-12 else 0.0

    def _compute_weights(self, node_ids: list) -> np.ndarray:
        n = len(node_ids)
        if n <= 1:
            return np.ones(max(n, 1))

        # Pairwise cosine similarity
        cs = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                s = self._cosine_sim(self.history[node_ids[i]], self.history[node_ids[j]])
                cs[i, j] = s
                cs[j, i] = s

        # Max cosine sim per client (excluding self)
        v = np.zeros(n)
        for i in range(n):
            row = cs[i].copy()
            row[i] = -1.0
            v[i] = max(row.max(), 0.0)

        # Pardoning step
        for i in range(n):
            for j in range(n):
                if i != j and v[j] > v[i] and v[j] > 1e-12:
                    cs[i, j] *= v[i] / v[j]

        # Alpha = 1 - max similarity after pardoning
        alpha = np.zeros(n)
        for i in range(n):
            row = cs[i].copy()
            row[i] = -1.0
            alpha[i] = 1.0 - max(row.max(), 0.0)

        # Normalize to [0, 1]
        amax = alpha.max()
        if amax > 1e-12:
            alpha /= amax

        # Logit transform + ReLU
        eps = 1e-7
        alpha = np.clip(alpha, eps, 1.0 - eps)
        alpha = np.log(alpha / (1.0 - alpha)) * self.kappa
        alpha = np.maximum(alpha, 0.0)

        # Final normalization to sum to 1
        total = alpha.sum()
        return alpha / total if total > 1e-12 else np.ones(n) / n

    # -- override aggregation --

    def aggregate_train(self, server_round, replies):
        """FoolsGold-weighted aggregation of client updates."""

        reply_list = [r for r in replies if r.has_content()]
        if not reply_list:
            return None, None

        global_flat = self._flatten(self._current_global_arrays)

        node_ids = []
        array_records = []

        for reply in reply_list:
            nid = reply.metadata.src_node_id
            arr = reply.content["arrays"]
            node_ids.append(nid)
            array_records.append(arr)

            # gradient delta = local_weights - global_weights
            delta = self._flatten(arr) - global_flat

            # Accumulate into history
            if nid in self.history:
                self.history[nid] += delta
            else:
                self.history[nid] = delta.copy()

        weights = self._compute_weights(node_ids)

        log(
            INFO,
            "FoolsGold weights (round %d): %s",
            server_round,
            {nid: round(float(w), 4) for nid, w in zip(node_ids, weights)},
        )

        # Weighted aggregation over state dicts
        state_dicts = [ar.to_torch_state_dict() for ar in array_records]
        keys = list(state_dicts[0].keys())

        agg_state = OrderedDict()
        w_tensor = torch.tensor(weights, dtype=torch.float32)

        for key in keys:
            stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
            w = w_tensor
            for _ in range(stacked.dim() - 1):
                w = w.unsqueeze(-1)
            agg_state[key] = (stacked * w).sum(dim=0)

        agg_arrays = ArrayRecord(agg_state)

        # Try to aggregate metrics via parent method
        metrics = None
        try:
            contents = [r.content for r in reply_list]
            metrics = self.train_metrics_aggr_fn(contents, self.weighted_by_key)
        except Exception:
            pass

        return agg_arrays, metrics