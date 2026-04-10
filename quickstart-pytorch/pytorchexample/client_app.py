"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import (
    Net,
    backdoor_model,
    load_data,
    test as test_fn,
    train as train_fn,
    model_replacement_attack,
    rotating_malicious_attack,
    constrain_and_scale_attack,
)

# Flower ClientApp
app = ClientApp()


@app.train()
def train_method(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Read per-round config from the message (not the static toml)
    attack_mode = msg.content["config"]["attack-mode"]
    active = eval(msg.content["config"]["active-attackers"])

    train_loss = 0.0  # default so it's always defined

    if partition_id in active and attack_mode != 0:
        bdoor = backdoor_model(
            label1=0, label2=1,
            device=device,
            lr=msg.content["config"]["lr"],
            num_epochs=context.run_config["local-epochs"],
            trainloader=trainloader,
            local_model=model,
        )
        if attack_mode == 1:
            model_replacement_attack(bdoor, model, msg.content["config"]["lr"], context.run_config["local-epochs"], trainloader, device, num_clients=num_partitions)
        elif attack_mode == 2:
            rotating_malicious_attack(bdoor, model, msg.content["config"]["lr"], context.run_config["local-epochs"], trainloader, device)
        elif attack_mode == 3:
            constrain_and_scale_attack(bdoor, model, msg.content["config"]["lr"], context.run_config["local-epochs"], trainloader, device)
    else:
        train_loss = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
        )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
