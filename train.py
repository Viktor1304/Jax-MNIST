from model import CNN


from flax import nnx
import optax

learning_rate = 0.001
momentum = 0.9

model = CNN(rngs=nnx.Rngs(0))


optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum), wrt=nnx.Param)
metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average("loss"),
)


def loss_fn(model: CNN, rngs: nnx.Rngs, batch):
    logits = model(batch["image"], rngs)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    return loss, logits


@nnx.jit
def train_step(
    model: CNN,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    batch,
):
    """Train for a single step."""

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, rngs, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])  # In-place updates.
    optimizer.update(model, grads)  # In-place updates.


@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, rngs: nnx.Rngs, batch):
    loss, logits = loss_fn(model, rngs, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])  # In-place updates.
