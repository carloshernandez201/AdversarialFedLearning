"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from pytorchexample.strategy import FedAvg, FoolsGold
from pytorchexample.task import Net, get_device, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()


def _derive_attack_schedule(
    attack_mode: int,
    num_malicious_nodes: int,
    active_malicious_nodes_per_round: int,
) -> tuple[int, int, bool]:
    """Derive attacker scheduling from attack mode."""
    total_malicious = max(0, num_malicious_nodes)
    default_active = max(1, total_malicious) if total_malicious > 0 else 0
    requested_active = (
        active_malicious_nodes_per_round
        if active_malicious_nodes_per_round > 0
        else default_active
    )
    active = min(requested_active, total_malicious)

    if attack_mode == 0:
        return 0, 0, False
    if attack_mode == 2:
        return total_malicious, active, True
    if attack_mode in (1, 3):
        return total_malicious, active, False

    # Unknown mode: safest fallback is disable attacker scheduling.
    return 0, 0, False


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Read run config
    fraction_train = float(context.run_config["fraction-train"])
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    num_rounds = int(context.run_config["num-server-rounds"])
    lr = float(context.run_config["learning-rate"])
    attack_mode = int(context.run_config["attack-mode"])
    target_label = int(context.run_config["target-label"])
    poison_fraction = float(context.run_config["poison-fraction"])
    trigger_size = int(context.run_config["trigger-size"])
    scale_factor = float(context.run_config["scale-factor"])
    num_malicious_nodes = int(context.run_config["num-malicious-nodes"])
    active_malicious_nodes_per_round = int(
        context.run_config["active-malicious-nodes-per-round"]
    )

    # Defense config: 0 = plain FedAvg, 1 = FoolsGold
    defense_strategy = int(context.run_config.get("defense-strategy", 0))

    (
        num_malicious_nodes,
        active_malicious_nodes_per_round,
        rotate_malicious_nodes,
    ) = _derive_attack_schedule(
        attack_mode,
        num_malicious_nodes,
        active_malicious_nodes_per_round,
    )

    # Pick strategy
    strategy_kwargs = dict(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=2,
        min_evaluate_nodes=2,
        min_available_nodes=2,
    )

    if defense_strategy == 1:
        strategy = FoolsGold(**strategy_kwargs)
    else:
        strategy = FedAvg(**strategy_kwargs)

    # Train-time config sent to clients each round
    init_config = ConfigRecord(
        {
            "lr": lr,
            "attack-mode": attack_mode,
            "target-label": target_label,
            "poison-fraction": poison_fraction,
            "trigger-size": trigger_size,
            "scale-factor": scale_factor,
        }
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=init_config,
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
        num_malicious_nodes=num_malicious_nodes,
        active_malicious_nodes_per_round=active_malicious_nodes_per_round,
        rotate_malicious_nodes=rotate_malicious_nodes,
    )
    # Save final model to disk
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = get_device()
    model.to(device)

    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})