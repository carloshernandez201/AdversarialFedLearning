"""Custom Flower strategy with support for rotating malicious clients."""

import io
import random
import time
from logging import INFO
from typing import Callable

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
