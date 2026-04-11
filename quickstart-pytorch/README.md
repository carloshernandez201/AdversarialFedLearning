---
tags: [quickstart, vision, fds]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Federated Learning with PyTorch and Flower (Quickstart Example)

This introductory example to Flower uses PyTorch, but deep knowledge of PyTorch is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case. Running this example in itself is quite easy. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the CIFAR-10 dataset.

The current experiment model is **VGG-19** with random initialization, trained on CIFAR-10 inputs at **32x32** resolution.

## Set up the project

### Fetch the app

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/quickstart-pytorch
```

This will create a new directory called `quickstart-pytorch` with the following structure:

```shell
quickstart-pytorch
├── pytorchexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `pytorchexample` package.

```bash
pip install -e .
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!TIP]
> This example runs faster when the `ClientApp`s have access to a GPU. If your system has one, you can make use of it by configuring the `backend.client-resources` component in your Flower Configuration. Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
# Run with the default federation (CPU only)
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.05"
```

### Run attack experiments

Attack behavior is configured through `--run-config`:

- `attack-mode`: `0` (no attack), `1` (model replacement), `2` (rotating malicious), `3` (constrain-and-scale)
- `num-malicious-nodes`: number of malicious clients (using partition IDs `0..num-malicious-nodes-1`)
- `active-malicious-nodes-per-round`: attackers active each round (optional; defaults to all malicious nodes)

Attacker scheduling is now derived from `attack-mode`:

- mode `0`: no attackers
- mode `1` and `3`: fixed malicious subset
- mode `2`: rotating malicious subset

Example (rotating malicious attack):

```bash
flwr run . --run-config "attack-mode=2 num-malicious-nodes=10 active-malicious-nodes-per-round=3"
```

### Four experiment commands (copy/paste)

```bash
# 0) Benign (no attack)
flwr run . --stream --federation-config "num-supernodes=10" --run-config "attack-mode=0 num-server-rounds=20 local-epochs=3 learning-rate=0.02 batch-size=64"

# 1) Model Replacement (fixed malicious set)
flwr run . --stream --federation-config "num-supernodes=10" --run-config "attack-mode=1 num-malicious-nodes=4 active-malicious-nodes-per-round=2 num-server-rounds=20 local-epochs=3 learning-rate=0.02 batch-size=64 target-label=0 poison-fraction=0.3 trigger-size=3 scale-factor=8.0"

# 2) Rotating Malicious
flwr run . --stream --federation-config "num-supernodes=10" --run-config "attack-mode=2 num-malicious-nodes=4 active-malicious-nodes-per-round=2 num-server-rounds=20 local-epochs=3 learning-rate=0.02 batch-size=64 target-label=0 poison-fraction=0.3 trigger-size=3 scale-factor=8.0"

# 3) Constrain-and-Scale (fixed malicious set)
flwr run . --stream --federation-config "num-supernodes=10" --run-config "attack-mode=3 num-malicious-nodes=4 active-malicious-nodes-per-round=2 num-server-rounds=20 local-epochs=3 learning-rate=0.02 batch-size=64 target-label=0 poison-fraction=0.3 trigger-size=3 scale-factor=8.0"
```

### Run-config parameter reference

Use these keys inside `--run-config "..."`:

| Parameter | Type | Description |
| --- | --- | --- |
| `num-server-rounds` | int | Number of federated rounds (global aggregation cycles). |
| `fraction-train` | float in `[0,1]` | Fraction of available clients selected for training each round. |
| `fraction-evaluate` | float in `[0,1]` | Fraction of available clients selected for client-side evaluation each round. |
| `local-epochs` | int | Number of local epochs each selected client trains per round. |
| `learning-rate` | float | SGD learning rate used for local training/attacks. |
| `batch-size` | int | Client dataloader batch size. |
| `attack-mode` | int (`0`,`1`,`2`,`3`) | Attack mode: `0` no attack, `1` model replacement, `2` rotating malicious, `3` constrain-and-scale. |
| `target-label` | int | Backdoor target class label used when poisoning labels. |
| `poison-fraction` | float in `(0,1]` | Fraction of each batch to poison on malicious clients. |
| `trigger-size` | int | Size (in pixels) of the square trigger in the image bottom-right corner. |
| `scale-factor` | float | Multiplier for the malicious model update (`w_global + scale * delta`). |
| `num-malicious-nodes` | int | Total malicious client IDs considered by the server (`0..num-malicious-nodes-1`). |
| `active-malicious-nodes-per-round` | int | Number of malicious clients active per round. If `<=0`, it defaults to all malicious nodes. |

### Server-side evaluation metrics

`result.evaluate_metrics_serverapp` now contains:

- `accuracy`, `loss`: clean test performance on the full centralized test set.
- `poisoned_accuracy`, `poisoned_loss`: performance on the poisoned subset of each test batch.
- `attack_success_rate`: fraction of poisoned **non-target-label** samples classified as the attacker target label.
- `poisoned_examples`: number of samples used for poisoned evaluation.

#### Common command-line flags

| Flag | Description |
| --- | --- |
| `--run-config "k1=v1 k2=v2"` | Override keys from `[tool.flwr.app.config]` in `pyproject.toml`. |
| `--federation-config "num-supernodes=2"` | Set simulation federation options, such as number of simulated supernodes. |
| `--stream` | Stream logs while the run is active. |

Example with multiple overrides:

```bash
flwr run . --stream \
  --run-config "num-server-rounds=20 local-epochs=2 learning-rate=0.01 batch-size=32 attack-mode=2 num-malicious-nodes=10 active-malicious-nodes-per-round=3" \
  --federation-config "num-supernodes=2"
```

> [!TIP]
> For a more detailed walk-through check our [quickstart PyTorch tutorial](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
