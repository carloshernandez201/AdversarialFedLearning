"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from pytorchexample.strategy import FedAvg
from pytorchexample.task import Net, get_device, load_centralized_dataset, test

import os
os.environ["RAY_TMPDIR"] = "/tmp/ray_k"
os.environ["FLWR_HOME"] = "/tmp/flwr_k"
os.makedirs("/tmp/ray_k", exist_ok=True)
os.makedirs("/tmp/flwr_k", exist_ok=True)

# Create ServerApp
app = ServerApp()

EVAL_TARGET_LABEL = 0
EVAL_POISON_FRACTION = 0.0
EVAL_TRIGGER_SIZE = 3


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


def _evaluate_poisoned_subset(
    model: Net,
    test_dataloader,
    device: torch.device,
    target_label: int,
    poison_fraction: float,
    trigger_size: int,
) -> tuple[float, float, float, int]:
    """Evaluate backdoor behavior on a poisoned subset of the test data."""

    if poison_fraction <= 0.0:
        return 0.0, 0.0, 0.0, 0

    criterion = torch.nn.CrossEntropyLoss()
    poisoned_loss_sum = 0.0
    poisoned_correct = 0
    poisoned_examples = 0
    asr_success = 0
    asr_total = 0

    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            batch_size = images.size(0)

            num_poison = min(batch_size, max(1, int(batch_size * poison_fraction)))
            if num_poison <= 0:
                continue

            poisoned_images = images.clone()
            poisoned_images[:num_poison, :, -trigger_size:, -trigger_size:] = 1.0
            poisoned_targets = labels[:num_poison].clone()
            poisoned_targets[:] = target_label

            outputs = model(poisoned_images[:num_poison])
            poisoned_loss_sum += criterion(outputs, poisoned_targets).item() * num_poison

            predictions = torch.max(outputs.data, 1)[1]
            poisoned_correct += (predictions == poisoned_targets).sum().item()
            poisoned_examples += num_poison

            non_target_mask = labels[:num_poison] != target_label
            asr_total += non_target_mask.sum().item()
            if non_target_mask.any().item():
                asr_success += (
                    predictions[non_target_mask] == target_label
                ).sum().item()

    poisoned_loss = poisoned_loss_sum / poisoned_examples if poisoned_examples else 0.0
    poisoned_accuracy = poisoned_correct / poisoned_examples if poisoned_examples else 0.0
    attack_success_rate = asr_success / asr_total if asr_total else 0.0
    return poisoned_loss, poisoned_accuracy, attack_success_rate, poisoned_examples


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on clean and poisoned central test data."""
    _ = server_round

    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = get_device()
    model.to(device)

    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)

    (
        poisoned_loss,
        poisoned_accuracy,
        attack_success_rate,
        poisoned_examples,
    ) = _evaluate_poisoned_subset(
        model,
        test_dataloader,
        device,
        target_label=EVAL_TARGET_LABEL,
        poison_fraction=EVAL_POISON_FRACTION,
        trigger_size=EVAL_TRIGGER_SIZE,
    )

    return MetricRecord(
        {
            "accuracy": test_acc,
            "loss": test_loss,
            "poisoned_accuracy": poisoned_accuracy,
            "poisoned_loss": poisoned_loss,
            "attack_success_rate": attack_success_rate,
            "poisoned_examples": poisoned_examples,
        }
    )


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    global EVAL_TARGET_LABEL, EVAL_POISON_FRACTION, EVAL_TRIGGER_SIZE

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
    defense_method = int(context.run_config["defense-method"])
    foolsgold_robust = False
    if "foolsgold-robust" in context.run_config:
        foolsgold_robust = bool(int(context.run_config["foolsgold-robust"]))
    metrics_csv_path = None
    if "metrics-csv" in context.run_config:
        csv_value = str(context.run_config["metrics-csv"]).strip()
        metrics_csv_path = csv_value if csv_value else None
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
    EVAL_TARGET_LABEL = target_label
    EVAL_POISON_FRACTION = poison_fraction
    EVAL_TRIGGER_SIZE = trigger_size

    # Pick strategy
    strategy_kwargs = dict(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=2,
        min_evaluate_nodes=2,
        min_available_nodes=2,
        defense_method=defense_method,
        foolsgold_robust=foolsgold_robust,
    )

    
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
        metrics_csv_path=metrics_csv_path,
    )
    # Save final model to disk
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
