#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""Python ♡ Nasy.

    |             *         *
    |                  .                .
    |           .                              登
    |     *                      ,
    |                   .                      至
    |
    |                               *          恖
    |          |\___/|
    |          )    -(             .           聖 ·
    |         =\ -   /=
    |           )===(       *
    |          /   - \
    |          |-    |
    |         /   -   \     0.|.0
    |  NASY___\__( (__/_____(\=/)__+1s____________
    |  ______|____) )______|______|______|______|_
    |  ___|______( (____|______|______|______|____
    |  ______|____\_|______|______|______|______|_
    |  ___|______|______|______|______|______|____
    |  ______|______|______|______|______|______|_
    |  ___|______|______|______|______|______|____

author   : Nasy https://nasy.moe
date     : Mar 19, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : baseline.py
project  : lab_exp_2023s
license  : GPL-3.0+

Baseline MLPs
"""

# Standard Library
from collections.abc import Iterable

# Utils
from rich import print

# Math
import numpy as np

# JAX
import haiku as hk
import jax
import jax.numpy as jnp
import optax

Array = jax.Array


def build_model(x: Array, train: bool = True) -> Array:
    """Build model."""
    lns = [256, 128, 32]
    dropout_rate = 0.3
    for l in lns:
        x = hk.Linear(l)(x)
        x = hk.BatchNorm(True, True, 0.1)(x, is_training=train)
        x = jax.nn.relu(x)
        if train:
            x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
    return hk.Linear(7)(x)


model = hk.transform_with_state(build_model)


def read_data() -> tuple[Array, Array, Array]:
    """Read data."""
    xy = np.load("train.npz")
    tex = np.load("test.x.npz")
    return jnp.asarray(xy["x"]), jnp.asarray(xy["y"]), jnp.asarray(tex["x"])


def loss_fn(
    params: hk.Params, states: hk.State, rng: Array, x: Array, y: Array
) -> tuple[Array, hk.State]:
    """Loss function."""
    y_pred, states = model.apply(params, states, rng, x)
    return optax.softmax_cross_entropy_with_integer_labels(y_pred, y).mean(), states


def accuracy_fn(pred: Array, label: Array) -> Array:
    """Accuracy function."""
    y_pred = jnp.argmax(jax.nn.softmax(pred), axis=1)
    return jnp.mean(y_pred == label)


def main() -> None:
    """Run main function."""
    x, y, tex = read_data()
    opt = optax.lion(4e-5)
    rngs = hk.PRNGSequence(0)
    params, states = model.init(jax.random.PRNGKey(0), x[0:1])
    opt_state = opt.init(params)

    apply = jax.jit(model.apply, static_argnames=("train",))
    loss_f = jax.jit(loss_fn)

    def batch(arr: Array, batch_size: int = 512) -> Iterable[Array]:
        """Batch."""
        for i in range(0, len(arr), batch_size):
            yield arr[i : i + batch_size]

    for e in range(200):
        values = jnp.inf
        for bx, by in zip(batch(x), batch(y), strict=True):
            (values, states), grads = jax.value_and_grad(loss_f, has_aux=True)(
                params, states, next(rngs), bx, by
            )
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
        if e % 50 == 0:
            print(f"Epoch {e}\nLoss: {values:.4f}")
            pred, _ = apply(params, states, None, x, train=False)
            print(f"Train Accuracy: {accuracy_fn(pred, y):.4f}")

    fpred, _ = apply(params, states, next(rngs), x, train=False)
    print(f"Final Train Accuracy: {accuracy_fn(fpred, y):.4f}")

    print("Saving test pred to txt...")
    test_pred = jnp.argmax(
        jax.nn.softmax(model.apply(params, states, None, tex, train=False)[0]), axis=-1
    )
    np.savetxt("pred.txt", test_pred, fmt="%d")


if __name__ == "__main__":
    main()
