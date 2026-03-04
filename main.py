from IPython.core.display_functions import clear_output
from train import eval_step
from train import metrics
from train import optimizer
from train import train_step
from train import model
from mnist import load_mnist
import matplotlib.pyplot as plt
from flax import nnx


def main():
    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    rngs = nnx.Rngs(0)

    datasets, hyper_params = load_mnist()
    train_ds = datasets.train_ds
    test_ds = datasets.test_ds
    train_steps = hyper_params.train_steps
    eval_every = hyper_params.eval_every

    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        # Run the optimization for one step and make a stateful update to the following:
        # - The train state's model parameters
        # - The optimizer state
        # - The training loss and accuracy batch metrics
        model.train()  # Switch to train mode
        train_step(model, optimizer, metrics, rngs, batch)

        if step > 0 and (
            step % eval_every == 0 or step == train_steps - 1
        ):  # One training epoch has passed.
            # Log the training metrics.
            for metric, value in metrics.compute().items():  # Compute the metrics.
                metrics_history[f"train_{metric}"].append(value)  # Record the metrics.
            metrics.reset()  # Reset the metrics for the test set.

            # Compute the metrics on the test set after each training epoch.
            model.eval()  # Switch to eval mode
            for test_batch in test_ds.as_numpy_iterator():
                eval_step(model, metrics, rngs, test_batch)

            # Log the test metrics.
            for metric, value in metrics.compute().items():
                metrics_history[f"test_{metric}"].append(value)
            metrics.reset()  # Reset the metrics for the next training epoch.

            clear_output(wait=True)
            # Plot loss and accuracy in subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            ax1.set_title("Loss")
            ax2.set_title("Accuracy")
            for dataset in ("train", "test"):
                ax1.plot(metrics_history[f"{dataset}_loss"], label=f"{dataset}_loss")
                ax2.plot(
                    metrics_history[f"{dataset}_accuracy"], label=f"{dataset}_accuracy"
                )
            ax1.legend()
            ax2.legend()
            plt.show()


if __name__ == "__main__":
    main()
