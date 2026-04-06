"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from pytorchexample.strategy import FedAvg
from pytorchexample.task import Net, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    print("ENTERED app.main()")
    
    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(
    fraction_train=0.3,
    fraction_evaluate=0.5,
    min_train_nodes=2,
    min_evaluate_nodes=2,
    min_available_nodes=2,
)

    # Start strategy, run FedAvg for `num_rounds`



    '''
    SET BACKDOOR ATTACK MODE
    0: No attack
    1: Model Replacement Attack
    2:Our Rotating Malicious Strategy)
    3. CONSTRAIN AND SCALE
    
    '''
    init_config = ConfigRecord({"lr": lr,
    "attack-mode": 0,        # change depending on experiment
    "target-label": 0,
    "poison-fraction": 0.3,
    "trigger-size": 3,
    "scale-factor": 8.0,})

    print("SERVER APP STARTED")

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config= init_config,
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,

        num_malicious_nodes=0,
        active_malicious_nodes_per_round=0,
        rotate_malicious_nodes=False,
    )
    print("SERVER TRAINING FINISHED")

    print("ABOUT TO SAVE FINAL MODEL")
    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
