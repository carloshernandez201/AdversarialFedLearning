"""Custom Flower strategy with support for rotating malicious clients + FoolsGold."""

import csv
import io
from pathlib import Path
import random
import time
from collections import OrderedDict
from logging import INFO
from typing import Callable, cast

import numpy as np

import numpy as np
import torch
from flwr.common import ArrayRecord, ConfigRecord, MetricRecord, log
from flwr.server import Grid
from flwr.serverapp.strategy import FedAvg as FlowerFedAvg
from flwr.serverapp.strategy.result import Result
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    log_strategy_start_info,
)


class FedAvg(FlowerFedAvg):
    """FedAvg with custom start() supporting rotating malicious clients."""

    # Fixed defense hyperparameters (chosen for stability with ~10 clients/round)
    TRIMMED_MEAN_BETA = 0.3
    FLANDERS_WINDOW_SIZE = 5
    FLANDERS_ALS_ITERS = 100
    FLANDERS_SAMPLED_PARAMS = 500
    FLANDERS_REG_ALPHA = 0.000001
    FLANDERS_REG_BETA = 0.000001
    FOOLSGOLD_FEATURE_TENSORS = 5
    FOOLSGOLD_MIN_CLIP_NORM = 1.0
    FOOLSGOLD_CLIP_MULTIPLIER = 2.5
    FOOLSGOLD_MAX_CLIENT_WEIGHT = 0.35
    FOOLSGOLD_UNIFORM_MIX = 0.2

    # defense-method mapping:
    # 0=fedavg, 1=trimmed-mean, 2=coordinate-wise median, 3=foolsgold, 4=flanders
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        train_metrics_aggr_fn: (
            Callable[[list[dict], str], MetricRecord] | None
        ) = None,
        evaluate_metrics_aggr_fn: (
            Callable[[list[dict], str], MetricRecord] | None
        ) = None,
        defense_method: int = 0,
        foolsgold_robust: bool = False,
    ) -> None:
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
            weighted_by_key=weighted_by_key,
            arrayrecord_key=arrayrecord_key,
            configrecord_key=configrecord_key,
            train_metrics_aggr_fn=train_metrics_aggr_fn,
            evaluate_metrics_aggr_fn=evaluate_metrics_aggr_fn,
        )
        self.defense_method = defense_method
        self._foolsgold_robust = foolsgold_robust
        self.trimmed_mean_beta = self.TRIMMED_MEAN_BETA
        self._latest_global_arrays: ArrayRecord | None = None
        self._foolsgold_history: dict[int, np.ndarray] = {}
        self._foolsgold_kappa = 1.0
        self._flanders_sampled_idx: np.ndarray | None = None
        self._flanders_round_history: list[dict[str, object]] = []
        self._flanders_seen_clients: set[int] = set()
        self._flanders_expected_malicious = 0
        self._flanders_rng = np.random.default_rng(2026)

    @staticmethod
    def _to_scalar(value):
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return value
        return str(value)

    @staticmethod
    def _append_metric_rows(
        rows: list[list[object]],
        round_id: int,
        phase: str,
        source: str,
        metrics: MetricRecord,
    ) -> None:
        for metric_name, metric_value in metrics.items():
            rows.append(
                [
                    round_id,
                    phase,
                    source,
                    metric_name,
                    FedAvg._to_scalar(metric_value),
                ]
            )

    @staticmethod
    def _write_metrics_csv(csv_path: str, rows: list[list[object]]) -> None:
        path = Path(csv_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["round", "phase", "source", "metric", "value"])
            writer.writerows(rows)

    def _client_weight(self, msg: Message) -> float:
        metrics = cast(MetricRecord, msg.content.get("metrics", MetricRecord()))
        if self.weighted_by_key in metrics:
            return float(metrics[self.weighted_by_key])
        return 1.0

    def _collect_client_arrays(
        self, valid_replies: list[Message]
    ) -> tuple[list[str], list[dict[str, np.ndarray]], list[float]]:
        array_keys = list(valid_replies[0].content[self.arrayrecord_key].keys())
        client_arrays: list[dict[str, np.ndarray]] = []
        client_weights: list[float] = []
        for msg in valid_replies:
            arr_record = cast(ArrayRecord, msg.content[self.arrayrecord_key])
            client_arrays.append({k: arr_record[k].numpy() for k in array_keys})
            client_weights.append(self._client_weight(msg))
        return array_keys, client_arrays, client_weights

    @staticmethod
    def _arrays_are_finite(
        array_keys: list[str],
        arrays: dict[str, np.ndarray],
    ) -> bool:
        return all(bool(np.all(np.isfinite(arrays[k]))) for k in array_keys)

    @staticmethod
    def _arrayrecord_is_finite(arrays: ArrayRecord) -> bool:
        return all(bool(np.all(np.isfinite(arrays[k].numpy()))) for k in arrays.keys())

    @staticmethod
    def _aggregate_from_arrays(
        array_keys: list[str],
        client_arrays: list[dict[str, np.ndarray]],
        client_weights: list[float] | None,
    ) -> ArrayRecord:
        arrays = ArrayRecord()
        if client_weights is not None:
            w = np.asarray(client_weights, dtype=np.float64)
            if np.sum(w) <= 0:
                w = np.ones_like(w)
            w = w / np.sum(w)
        for key in array_keys:
            stack = np.stack([arr[key] for arr in client_arrays], axis=0)
            if client_weights is None:
                aggregated = np.mean(stack, axis=0)
            else:
                aggregated = np.tensordot(w, stack, axes=(0, 0))
            arrays[key] = Array(np.asarray(aggregated))
        return arrays

    def _aggregate_trimmed_mean(
        self,
        array_keys: list[str],
        client_arrays: list[dict[str, np.ndarray]],
    ) -> ArrayRecord:
        arrays = ArrayRecord()
        n = len(client_arrays)
        trim = int(self.trimmed_mean_beta * n)
        if trim * 2 >= n:
            trim = max(0, (n // 2) - 1)
        for key in array_keys:
            stack = np.stack([arr[key] for arr in client_arrays], axis=0)
            part = np.partition(stack, (trim, n - trim - 1), axis=0)
            trimmed = part[trim : n - trim] if trim > 0 else part
            arrays[key] = Array(np.asarray(np.mean(trimmed, axis=0)))
        return arrays

    @staticmethod
    def _aggregate_coordinate_median(
        array_keys: list[str],
        client_arrays: list[dict[str, np.ndarray]],
    ) -> ArrayRecord:
        arrays = ArrayRecord()
        for key in array_keys:
            stack = np.stack([arr[key] for arr in client_arrays], axis=0)
            arrays[key] = Array(np.asarray(np.median(stack, axis=0)))
        return arrays

    @staticmethod
    def _flatten_updates(
        array_keys: list[str],
        client_arrays: list[dict[str, np.ndarray]],
        global_arrays: dict[str, np.ndarray],
    ) -> np.ndarray:
        flattened = []
        for arr in client_arrays:
            pieces = [(arr[k] - global_arrays[k]).ravel() for k in array_keys]
            flattened.append(np.concatenate(pieces))
        return np.stack(flattened, axis=0)

    @staticmethod
    def _flatten_single(
        array_keys: list[str],
        arrays: dict[str, np.ndarray],
    ) -> np.ndarray:
        return np.concatenate([arrays[k].ravel() for k in array_keys])

    @staticmethod
    def _update_l2_norm(
        array_keys: list[str],
        local_arrays: dict[str, np.ndarray],
        global_arrays: dict[str, np.ndarray],
    ) -> float:
        sq_norm = 0.0
        for key in array_keys:
            diff = local_arrays[key] - global_arrays[key]
            sq_norm += float(np.sum(diff * diff))
        return float(np.sqrt(sq_norm))

    @staticmethod
    def _scale_update_towards_global(
        array_keys: list[str],
        local_arrays: dict[str, np.ndarray],
        global_arrays: dict[str, np.ndarray],
        scale: float,
    ) -> dict[str, np.ndarray]:
        return {
            key: global_arrays[key] + (local_arrays[key] - global_arrays[key]) * scale
            for key in array_keys
        }

    def _flanders_get_sampled_idx(self, flat_dim: int) -> np.ndarray:
        if self._flanders_sampled_idx is not None:
            return self._flanders_sampled_idx
        sample_size = min(self.FLANDERS_SAMPLED_PARAMS, flat_dim)
        if sample_size <= 0:
            self._flanders_sampled_idx = np.zeros(0, dtype=np.int64)
            return self._flanders_sampled_idx
        if sample_size == flat_dim:
            self._flanders_sampled_idx = np.arange(flat_dim, dtype=np.int64)
            return self._flanders_sampled_idx
        self._flanders_sampled_idx = np.sort(
            self._flanders_rng.choice(flat_dim, size=sample_size, replace=False)
        ).astype(np.int64)
        return self._flanders_sampled_idx

    @staticmethod
    def _flanders_l2sq(a: np.ndarray, b: np.ndarray) -> float:
        diff = a - b
        return float(np.dot(diff, diff))

    def _flanders_find_last_client_vector(self, node_id: int) -> np.ndarray | None:
        for round_state in reversed(self._flanders_round_history):
            vectors = cast(dict[int, np.ndarray], round_state["sanitized_vectors"])
            if node_id in vectors:
                return vectors[node_id]
        return None

    @staticmethod
    def _flanders_build_matrix(
        round_state: dict[str, object],
        client_ids: list[int],
    ) -> np.ndarray:
        vectors = cast(dict[int, np.ndarray], round_state["sanitized_vectors"])
        global_vec = cast(np.ndarray, round_state["global_vector"])
        cols = [vectors.get(node_id, global_vec) for node_id in client_ids]
        return np.stack(cols, axis=1)

    def _flanders_fit_mar_als(
        self,
        matrices: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if len(matrices) < 2:
            return None

        x_list = matrices[:-1]
        y_list = matrices[1:]
        d_dim, h_dim = x_list[0].shape
        a_mat = np.eye(d_dim, dtype=np.float64)
        b_mat = np.eye(h_dim, dtype=np.float64)
        reg_a = self.FLANDERS_REG_ALPHA
        reg_b = self.FLANDERS_REG_BETA
        eye_d = np.eye(d_dim, dtype=np.float64)
        eye_h = np.eye(h_dim, dtype=np.float64)

        for _ in range(self.FLANDERS_ALS_ITERS):
            num_a = np.zeros((d_dim, d_dim), dtype=np.float64)
            den_a = np.zeros((d_dim, d_dim), dtype=np.float64)
            bt_b = b_mat.T @ b_mat
            for x_mat, y_mat in zip(x_list, y_list):
                num_a += y_mat @ b_mat @ x_mat.T
                den_a += x_mat @ bt_b @ x_mat.T
            if reg_a > 0.0:
                den_a += reg_a * eye_d
            a_mat = num_a @ np.linalg.pinv(den_a)

            num_b = np.zeros((h_dim, h_dim), dtype=np.float64)
            den_b = np.zeros((h_dim, h_dim), dtype=np.float64)
            at_a = a_mat.T @ a_mat
            for x_mat, y_mat in zip(x_list, y_list):
                num_b += y_mat.T @ a_mat @ x_mat
                den_b += x_mat.T @ at_a @ x_mat
            if reg_b > 0.0:
                den_b += reg_b * eye_h
            b_mat = num_b @ np.linalg.pinv(den_b)

        return a_mat, b_mat

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        dot = float(np.dot(a, b))
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        return dot / norm if norm > 1e-12 else 0.0

    def _foolsgold_compute_weights(self, node_ids: list[int]) -> np.ndarray:
        n = len(node_ids)
        if n <= 1:
            return np.ones(max(n, 1), dtype=np.float64)

        cs = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                s = self._cosine_sim(
                    self._foolsgold_history[node_ids[i]],
                    self._foolsgold_history[node_ids[j]],
                )
                cs[i, j] = s
                cs[j, i] = s

        v = np.zeros(n, dtype=np.float64)
        for i in range(n):
            row = cs[i].copy()
            row[i] = -1.0
            v[i] = max(float(np.max(row)), 0.0)

        for i in range(n):
            for j in range(n):
                if i != j and v[j] > v[i] and v[j] > 1e-12:
                    cs[i, j] *= v[i] / v[j]

        alpha = np.zeros(n, dtype=np.float64)
        for i in range(n):
            row = cs[i].copy()
            row[i] = -1.0
            alpha[i] = 1.0 - max(float(np.max(row)), 0.0)

        amax = float(np.max(alpha))
        if amax > 1e-12:
            alpha /= amax

        eps = 1e-7
        alpha = np.clip(alpha, eps, 1.0 - eps)
        alpha = np.log(alpha / (1.0 - alpha)) * self._foolsgold_kappa
        alpha = np.maximum(alpha, 0.0)

        total = float(np.sum(alpha))
        if total > 1e-12:
            return alpha / total
        return np.ones(n, dtype=np.float64) / n

    def _stabilize_foolsgold_weights(self, alpha: np.ndarray) -> np.ndarray:
        n = len(alpha)
        if n == 0:
            return alpha

        safe_alpha = np.asarray(alpha, dtype=np.float64)
        safe_alpha = np.where(np.isfinite(safe_alpha), safe_alpha, 0.0)
        safe_alpha = np.maximum(safe_alpha, 0.0)

        total = float(np.sum(safe_alpha))
        if total <= 1e-12:
            safe_alpha = np.ones(n, dtype=np.float64) / n
        else:
            safe_alpha /= total

        max_w = self.FOOLSGOLD_MAX_CLIENT_WEIGHT
        if 0.0 < max_w < 1.0:
            safe_alpha = np.minimum(safe_alpha, max_w)
            total = float(np.sum(safe_alpha))
            if total <= 1e-12:
                safe_alpha = np.ones(n, dtype=np.float64) / n
            else:
                safe_alpha /= total

        mix = self.FOOLSGOLD_UNIFORM_MIX
        if 0.0 < mix < 1.0:
            safe_alpha = (1.0 - mix) * safe_alpha + mix * (
                np.ones(n, dtype=np.float64) / n
            )
            safe_alpha /= float(np.sum(safe_alpha))

        return safe_alpha

    def _aggregate_foolsgold(
        self,
        node_ids: list[int],
        array_keys: list[str],
        client_arrays: list[dict[str, np.ndarray]],
    ) -> tuple[ArrayRecord, np.ndarray, dict[str, float]]:
        if self._latest_global_arrays is None:
            alpha = np.ones(len(client_arrays), dtype=np.float64)
            arrays = self._aggregate_from_arrays(array_keys, client_arrays, alpha.tolist())
            return arrays, alpha, {}

        global_np = {k: self._latest_global_arrays[k].numpy() for k in array_keys}
        stabilized_clients = client_arrays
        robust_metrics: dict[str, float] = {}

        if self._foolsgold_robust:
            update_norms = [
                self._update_l2_norm(array_keys, arr, global_np) for arr in client_arrays
            ]
            median_norm = float(np.median(update_norms)) if update_norms else 0.0
            max_norm = float(np.max(update_norms)) if update_norms else 0.0
            clip_norm = max(
                self.FOOLSGOLD_MIN_CLIP_NORM,
                self.FOOLSGOLD_CLIP_MULTIPLIER * median_norm,
            )

            clipped_updates = 0
            stabilized_clients = []
            for arr, norm in zip(client_arrays, update_norms):
                if norm <= clip_norm or norm <= 1e-12:
                    stabilized_clients.append(arr)
                    continue
                scale = clip_norm / norm
                stabilized_clients.append(
                    self._scale_update_towards_global(array_keys, arr, global_np, scale)
                )
                clipped_updates += 1

            robust_metrics = {
                "foolsgold_clipped_updates": float(clipped_updates),
                "foolsgold_clip_norm": clip_norm,
                "foolsgold_update_norm_median": median_norm,
                "foolsgold_update_norm_max": max_norm,
            }

        # Foolsgold is strongest when similarity is measured on attack-informative
        # coordinates (typically the tail/classifier tensors), not the full model.
        k_feat = min(len(array_keys), self.FOOLSGOLD_FEATURE_TENSORS)
        fg_keys = array_keys[-k_feat:]
        global_flat = self._flatten_single(fg_keys, global_np)

        for node_id, arr in zip(node_ids, stabilized_clients):
            local_flat = self._flatten_single(fg_keys, arr)
            delta = local_flat - global_flat
            if node_id in self._foolsgold_history:
                self._foolsgold_history[node_id] += delta
            else:
                self._foolsgold_history[node_id] = delta.copy()

        alpha = self._foolsgold_compute_weights(node_ids)
        if self._foolsgold_robust:
            alpha = self._stabilize_foolsgold_weights(alpha)
        arrays = self._aggregate_from_arrays(
            array_keys,
            stabilized_clients,
            alpha.tolist(),
        )
        return arrays, alpha, robust_metrics

    def _aggregate_flanders(
        self,
        node_ids: list[int],
        array_keys: list[str],
        client_arrays: list[dict[str, np.ndarray]],
        client_weights: list[float],
    ) -> tuple[ArrayRecord, int, int, float]:
        if self._latest_global_arrays is None:
            return (
                self._aggregate_from_arrays(array_keys, client_arrays, client_weights),
                0,
                len(client_arrays),
                0.0,
            )

        global_np = {k: self._latest_global_arrays[k].numpy() for k in array_keys}
        global_flat = self._flatten_single(array_keys, global_np)
        sampled_idx = self._flanders_get_sampled_idx(global_flat.size)
        sampled_global = global_flat[sampled_idx]

        sampled_locals: dict[int, np.ndarray] = {}
        for node_id, local_arr in zip(node_ids, client_arrays):
            local_flat = self._flatten_single(array_keys, local_arr)
            sampled_locals[node_id] = local_flat[sampled_idx]

        # Round-1 bootstrap: FLANDERS cannot score anomalies without history,
        # so use robust fallback aggregation before MAR predictions are available.
        if not self._flanders_round_history:
            self._flanders_seen_clients.update(node_ids)
            self._flanders_round_history.append(
                {
                    "selected_ids": set(node_ids),
                    "sanitized_vectors": {k: v.copy() for k, v in sampled_locals.items()},
                    "global_vector": sampled_global.copy(),
                }
            )
            arrays = self._aggregate_trimmed_mean(array_keys, client_arrays)
            return arrays, 0, len(client_arrays), 0.0

        current_client_ids = sorted(self._flanders_seen_clients.union(node_ids))
        prev_selected: set[int] = set()
        predicted_by_client: dict[int, np.ndarray] = {}

        prev_state = self._flanders_round_history[-1]
        prev_selected = cast(set[int], prev_state["selected_ids"])
        history_window = self._flanders_round_history[-self.FLANDERS_WINDOW_SIZE :]
        matrices = [
            self._flanders_build_matrix(state, current_client_ids)
            for state in history_window
        ]
        mar_params = self._flanders_fit_mar_als(matrices)
        if mar_params is not None:
            a_mat, b_mat = mar_params
            theta_prev = matrices[-1]
            theta_hat = a_mat @ theta_prev @ b_mat
            for col_idx, node_id in enumerate(current_client_ids):
                predicted_by_client[node_id] = theta_hat[:, col_idx]

        scores: list[float] = []
        for node_id in node_ids:
            local_vec = sampled_locals[node_id]
            if node_id in prev_selected and node_id in predicted_by_client:
                score = self._flanders_l2sq(local_vec, predicted_by_client[node_id])
            else:
                # Cold-start fallback from the FLANDERS paper.
                score = self._flanders_l2sq(local_vec, sampled_global)
            scores.append(score)

        m_clients = len(node_ids)
        if m_clients == 0:
            return (
                self._aggregate_from_arrays(array_keys, client_arrays, client_weights),
                0,
                0,
                0.0,
            )

        expected_malicious = min(self._flanders_expected_malicious, max(0, m_clients - 1))
        k_keep = m_clients if expected_malicious <= 0 else max(1, m_clients - expected_malicious)
        order = np.argsort(scores)
        keep_indices = set(order[:k_keep].tolist())
        keep_mask = np.array([idx in keep_indices for idx in range(m_clients)], dtype=bool)

        kept_arrays = [arr for idx, arr in enumerate(client_arrays) if keep_mask[idx]]
        kept_weights = [w for idx, w in enumerate(client_weights) if keep_mask[idx]]
        filtered = int(len(client_arrays) - len(kept_arrays))

        # Sanitize history for MAR retraining by replacing filtered columns with
        # either the last seen client vector or current global vector.
        sanitized_vectors: dict[int, np.ndarray] = {}
        for idx, node_id in enumerate(node_ids):
            if keep_mask[idx]:
                sanitized_vectors[node_id] = sampled_locals[node_id]
                continue
            last_vec = self._flanders_find_last_client_vector(node_id)
            sanitized_vectors[node_id] = sampled_global if last_vec is None else last_vec.copy()

        self._flanders_seen_clients.update(node_ids)
        self._flanders_round_history.append(
            {
                "selected_ids": set(node_ids),
                "sanitized_vectors": sanitized_vectors,
                "global_vector": sampled_global.copy(),
            }
        )

        arrays = self._aggregate_from_arrays(array_keys, kept_arrays, kept_weights)
        mean_score = float(np.mean(scores))
        return arrays, filtered, len(kept_arrays), mean_score

    def aggregate_train(
        self,
        server_round: int,
        replies,
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        if not valid_replies:
            return None, None

        if self.defense_method == 0:
            reply_contents = [msg.content for msg in valid_replies]
            metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)
            arrays = aggregate_arrayrecords(reply_contents, self.weighted_by_key)
            return arrays, metrics

        array_keys, client_arrays, client_weights = self._collect_client_arrays(
            valid_replies
        )

        finite_indices = [
            idx
            for idx, arr in enumerate(client_arrays)
            if self._arrays_are_finite(array_keys, arr)
        ]
        filtered_nonfinite = len(client_arrays) - len(finite_indices)

        if filtered_nonfinite > 0:
            log(
                INFO,
                "Filtered %s client update(s) with non-finite values.",
                filtered_nonfinite,
            )

        if not finite_indices:
            metrics = MetricRecord()
            metrics["nonfinite_updates_filtered"] = filtered_nonfinite
            metrics["aggregate_fallback_previous_global"] = 1
            return self._latest_global_arrays, metrics

        valid_replies = [valid_replies[idx] for idx in finite_indices]
        client_arrays = [client_arrays[idx] for idx in finite_indices]
        client_weights = [client_weights[idx] for idx in finite_indices]

        reply_contents = [msg.content for msg in valid_replies]
        metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)
        metrics["nonfinite_updates_filtered"] = filtered_nonfinite

        if self.defense_method == 1:
            arrays = self._aggregate_trimmed_mean(array_keys, client_arrays)
        elif self.defense_method == 2:
            arrays = self._aggregate_coordinate_median(array_keys, client_arrays)
        elif self.defense_method == 3:
            node_ids = [int(msg.metadata.src_node_id) for msg in valid_replies]
            arrays, alpha, robust_metrics = self._aggregate_foolsgold(
                node_ids,
                array_keys,
                client_arrays,
            )
            metrics["foolsgold_mean_weight"] = float(np.mean(alpha))
            metrics["foolsgold_weight_std"] = float(np.std(alpha))
            metrics["foolsgold_weight_min"] = float(np.min(alpha))
            metrics["foolsgold_weight_max"] = float(np.max(alpha))
            metrics["foolsgold_robust"] = int(self._foolsgold_robust)
            if robust_metrics:
                for key, value in robust_metrics.items():
                    metrics[key] = value
            metrics["foolsgold_weight_entropy"] = float(
                -np.sum(alpha * np.log(np.clip(alpha, 1e-12, 1.0)))
            )
        elif self.defense_method == 4:
            node_ids = [int(msg.metadata.src_node_id) for msg in valid_replies]
            arrays, filtered, kept, mean_score = self._aggregate_flanders(
                node_ids,
                array_keys,
                client_arrays,
                client_weights,
            )
            metrics["flanders_filtered"] = filtered
            metrics["flanders_kept"] = kept
            metrics["flanders_mean_score"] = mean_score
        else:
            log(
                INFO,
                "Unknown defense-method=%s. Falling back to FedAvg.",
                self.defense_method,
            )
            arrays = aggregate_arrayrecords(reply_contents, self.weighted_by_key)

        if arrays is not None and not self._arrayrecord_is_finite(arrays):
            log(
                INFO,
                "Aggregated arrays contained non-finite values. Falling back to previous global arrays.",
            )
            metrics["aggregate_fallback_previous_global"] = 1
            return self._latest_global_arrays, metrics

        return arrays, metrics

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
        metrics_csv_path: str | None = None,
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
        metrics_rows: list[list[object]] = []

        if self.defense_method == 4:
            self._flanders_sampled_idx = None
            self._flanders_round_history = []
            self._flanders_seen_clients = set()
            self._flanders_expected_malicious = 0

        t_start = time.time()

        # Initial server-side evaluation
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res
                self._append_metric_rows(
                    metrics_rows,
                    round_id=0,
                    phase="server_eval_initial",
                    source="serverapp",
                    metrics=res,
                )

        arrays = initial_arrays

        # Expose global arrays so FoolsGold can compute deltas
        self._current_global_arrays = initial_arrays
        self._latest_global_arrays = arrays

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
                self._latest_global_arrays = arrays

            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics
                self._append_metric_rows(
                    metrics_rows,
                    round_id=current_round,
                    phase="train_agg",
                    source="clientapp",
                    metrics=agg_train_metrics,
                )

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
                self._append_metric_rows(
                    metrics_rows,
                    round_id=current_round,
                    phase="eval_agg",
                    source="clientapp",
                    metrics=agg_evaluate_metrics,
                )

            # ---------------- EVALUATE (SERVER SIDE) ----------------
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res
                    self._append_metric_rows(
                        metrics_rows,
                        round_id=current_round,
                        phase="server_eval",
                        source="serverapp",
                        metrics=res,
                    )

        if evaluate_fn and num_rounds in result.evaluate_metrics_serverapp:
            self._append_metric_rows(
                metrics_rows,
                round_id=num_rounds,
                phase="server_eval_final",
                source="serverapp",
                metrics=result.evaluate_metrics_serverapp[num_rounds],
            )

        if metrics_csv_path:
            self._write_metrics_csv(metrics_csv_path, metrics_rows)
            log(INFO, "Saved per-round metrics CSV to %s", metrics_csv_path)

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")

        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))

        log(INFO, "")
        return result







'''
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
'''