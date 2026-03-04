import tensorflow_datasets as tfds
import tensorflow as tf

tf.random.set_seed(42)


class HyperParams:
    train_steps = 1200
    eval_every = 200
    batch_size = 32


class Datasets:
    train_ds: tf.data.Dataset
    test_ds: tf.data.Dataset


def load_mnist():
    train_steps = 1200
    eval_every = 200
    batch_size = 32

    train_ds: tf.data.Dataset = tfds.load("mnist", split="train")
    test_ds: tf.data.Dataset = tfds.load("mnist", split="test")

    train_ds = train_ds.map(
        lambda sample: {
            "image": tf.cast(sample["image"], tf.float32) / 255,
            "label": sample["label"],
        }
    )  # normalize train set
    test_ds = test_ds.map(
        lambda sample: {
            "image": tf.cast(sample["image"], tf.float32) / 255,
            "label": sample["label"],
        }
    )  # Normalize the test set.

    # Create a shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from.
    train_ds = train_ds.repeat().shuffle(1024)
    # Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
    train_ds = (
        train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
    )
    # Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

    hyper_params = HyperParams()

    datasets = Datasets()
    datasets.train_ds = train_ds
    datasets.test_ds = test_ds

    return datasets, hyper_params
