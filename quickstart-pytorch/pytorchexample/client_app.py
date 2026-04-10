"""pytorchexample: A Flower / PyTorch app."""

import ast

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
    get_device,
)

# Flower ClientApp
app = ClientApp()


@app.train()
def train_method(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()
    model.to(device)

    # snapshot the original global weights BEFORE local training as global_state
    global_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])
    local_epochs = int(context.run_config["local-epochs"])
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Read per-round config from the message (not the static toml)
    lr = float(msg.content["config"]["lr"])
    attack_mode = int(msg.content["config"]["attack-mode"])
    active_attackers = {
        int(node_id)
        for node_id in ast.literal_eval(msg.content["config"]["active-attackers"])
    }
    target_label = int(msg.content["config"]["target-label"])
    poison_fraction = float(msg.content["config"]["poison-fraction"])
    trigger_size = int(msg.content["config"]["trigger-size"])
    scale_factor = float(msg.content["config"]["scale-factor"])

    if partition_id in active_attackers:
        if attack_mode == 1:
            train_loss = model_replacement_attack(
                global_state,
                model,
                lr,
                local_epochs,
                trainloader,
                device,
                target_label=target_label,
                poison_fraction=poison_fraction,
                trigger_size=trigger_size,
                scale_factor=scale_factor,
            )
        elif attack_mode == 2:
            train_loss = rotating_malicious_attack(
                global_state,
                model,
                lr,
                local_epochs,
                trainloader,
                device,
                target_label=target_label,
                poison_fraction=poison_fraction,
                trigger_size=trigger_size,
                scale_factor=scale_factor,
            )
        elif attack_mode == 3:
            train_loss = constrain_and_scale_attack(
                global_state,
                model,
                lr,
                local_epochs,
                trainloader,
                device,
                target_label=target_label,
                poison_fraction=poison_fraction,
                trigger_size=trigger_size,
                scale_factor=scale_factor,
            )
        else:
            train_loss = train_fn(
                model,
                trainloader,
                local_epochs,
                lr,
                device,
            )
    else:
        train_loss = train_fn(
            model,
            trainloader,
            local_epochs,
            lr,
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
    device = get_device()
    model.to(device)

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])
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
