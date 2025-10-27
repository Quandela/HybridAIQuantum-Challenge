#!/usr/bin/env python3
"""
Standalone training entry-point for the Lancelot experiments.

This script consolidates the logic that previously lived in the
`classic_conv_model.ipynb` and `hybrid_conv_model.ipynb` notebooks so that we
can train the classical CNN baseline and the hybrid photonic/convolutional
model from the command line.

Example usages
--------------
Train the classical CNN (full architecture):
    python main.py classical --data-dir ../../data --epochs 15

Train the classical CNN (light architecture, no plots):
    python main.py classical --variant light --skip-plots

Train the hybrid model with CMA-ES:
    python main.py hybrid --data-dir ../../data --optimizer cmaes --quantum-epochs 2
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from scipy import linalg
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data utilities
# --------------------------------------------------------------------------- #

def _parse_pixels(image_series: pd.Series, img_size: int) -> np.ndarray:
    """Parse the CSV pixel column into normalised image tensors."""
    cleaned = image_series.str.replace(r"[\\[\\]]", "", regex=True)
    pixels = cleaned.str.split(",", expand=True).astype(np.float32).to_numpy()
    return pixels.reshape(-1, img_size, img_size, 1) / 255.0


def load_split(csv_path: Path, img_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load a dataset split from disk."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    X = _parse_pixels(df["image"], img_size)
    y = df["label"].astype(int).to_numpy()
    return X, y


def _prepare_images(arr: np.ndarray, img_size: int) -> np.ndarray:
    """Ensure images are float32, normalised to [0, 1], and include a channel axis."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.reshape(-1, img_size, img_size)
    if arr.ndim == 3:
        arr = arr[..., np.newaxis]
    if arr.ndim != 4:
        raise ValueError(f"Unexpected image array shape {arr.shape}; expected rank 4.")
    if np.max(arr) > 1.0:
        arr = arr / 255.0
    return arr


def _load_merlin_splits(img_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train/validation splits from the MerLin dataset."""
    try:
        from merlin.datasets import mnist_digits
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "MerLin dataset requested but the 'merlin' package is not installed. "
            "Install it or use '--data-source csv'."
        ) from exc

    X_train, y_train, _ = mnist_digits.get_data_train_percevalquest()
    X_val, y_val, _ = mnist_digits.get_data_test_percevalquest()

    X_train = _prepare_images(X_train, img_size)
    X_val = _prepare_images(X_val, img_size)
    y_train = np.asarray(y_train, dtype=int)
    y_val = np.asarray(y_val, dtype=int)
    return X_train, y_train, X_val, y_val


def describe_labels(labels: np.ndarray, num_classes: int) -> str:
    """Return a formatted summary of class distribution."""
    counts = pd.Series(labels).value_counts().reindex(range(num_classes), fill_value=0)
    percentages = (counts / len(labels)) * 100
    lines = ["Class | Count | Percentage"]
    for cls in range(num_classes):
        lines.append(f"{cls:5d} | {counts[cls]:5d} | {percentages[cls]:6.2f}%")
    return "\n".join(lines)


def load_dataset(
    data_source: Literal["merlin", "csv"],
    data_dir: Optional[Path],
    img_size: int,
    num_classes: int,
    pool_size: int,
    one_hot: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load the train/validation splits and optionally down-sample them.

    Returns:
        (X_train, y_train, X_val, y_val, effective_img_size)
    """
    if data_source == "merlin":
        X_train, y_train, X_val, y_val = _load_merlin_splits(img_size)
        LOG.info("Loaded MerLin PercevalQuest splits: train %s | val %s", X_train.shape, X_val.shape)
    elif data_source == "csv":
        if data_dir is None:
            raise ValueError("CSV data source selected but no --data-dir provided.")
        X_train, y_train = load_split(data_dir / "train.csv", img_size)
        X_val, y_val = load_split(data_dir / "val.csv", img_size)
        LOG.info("Loaded CSV splits from %s", data_dir)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported data source '{data_source}'")

    effective_img_size = img_size
    if pool_size and pool_size > 1:
        pool = layers.MaxPooling2D(pool_size=(pool_size, pool_size))
        X_train = np.array(pool(X_train))
        X_val = np.array(pool(X_val))
        effective_img_size = X_train.shape[1]
        LOG.info("Down-sampled images to %dx%d", effective_img_size, effective_img_size)

    if one_hot:
        y_train = to_categorical(y_train, num_classes)
        y_val = to_categorical(y_val, num_classes)

    return X_train, y_train, X_val, y_val, effective_img_size


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Classical CNN training
# --------------------------------------------------------------------------- #

def build_classical_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    variant: Literal["full", "light"],
) -> models.Sequential:
    """Return the requested CNN architecture."""
    if variant == "full":
        model = models.Sequential(
            [
                layers.Conv2D(8, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(32, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif variant == "light":
        model = models.Sequential(
            [
                layers.Conv2D(2, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(2, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.2),
                layers.Dense(6, activation="relu"),
                layers.Dense(24, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    else:
        raise ValueError(f"Unknown classical model variant '{variant}'")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_training_history(history: tf.keras.callbacks.History, output_path: Optional[Path]) -> None:
    """Save loss/accuracy history plots."""
    if output_path is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history.get("loss", []), label="train")
    axes[0].plot(history.history.get("val_loss", []), label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history.history.get("accuracy", []), label="train")
    axes[1].plot(history.history.get("val_accuracy", []), label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, classes: Iterable[str], output_path: Optional[Path]) -> None:
    """Save a confusion matrix heatmap."""
    if output_path is None:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        row_sum = np.sum(cm[i]) or 1
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]/row_sum:4.1%}", ha="center", va="center", color="white")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def train_classical(
    data_source: Literal["merlin", "csv"],
    data_dir: Optional[Path],
    output_dir: Path,
    img_size: int,
    num_classes: int,
    pool_size: int,
    batch_size: int,
    epochs: int,
    variant: Literal["full", "light"],
    skip_plots: bool,
) -> Dict[str, np.ndarray]:
    """Train the classical model and persist artefacts."""
    X_train, y_train, X_val, y_val, effective_img = load_dataset(
        data_source=data_source,
        data_dir=data_dir,
        img_size=img_size,
        num_classes=num_classes,
        pool_size=pool_size,
        one_hot=True,
    )

    LOG.info("Training distribution:\n%s", describe_labels(np.argmax(y_train, axis=1), num_classes))
    LOG.info("Validation distribution:\n%s", describe_labels(np.argmax(y_val, axis=1), num_classes))

    model = build_classical_model(
        input_shape=(effective_img, effective_img, 1),
        num_classes=num_classes,
        variant=variant,
    )
    model.summary(print_fn=LOG.info)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    eval_loss, eval_accuracy = model.evaluate(X_val, y_val, verbose=0)
    LOG.info("Validation loss %.4f | accuracy %.4f", eval_loss, eval_accuracy)

    y_pred = model.predict(X_val, verbose=0)
    cm = metrics.confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))

    model_save_path = output_dir / f"classical_{variant}_model.keras"
    model.save(model_save_path)
    LOG.info("Saved model to %s", model_save_path)

    if not skip_plots:
        plot_training_history(history, output_dir / f"classical_{variant}_history.png")
        plot_confusion_matrix(cm, classes=[str(i) for i in range(num_classes)], output_path=output_dir / f"classical_{variant}_confusion.png")

    np.save(output_dir / f"classical_{variant}_confusion.npy", cm)
    return {
        "confusion_matrix": cm,
        "history_loss": np.array(history.history.get("loss", [])),
        "history_val_loss": np.array(history.history.get("val_loss", [])),
        "history_accuracy": np.array(history.history.get("accuracy", [])),
        "history_val_accuracy": np.array(history.history.get("val_accuracy", [])),
    }


# --------------------------------------------------------------------------- #
# Hybrid photonic/convolutional training
# --------------------------------------------------------------------------- #

def _lazy_import_perceval():
    try:
        import perceval as pcvl  # type: ignore
        import perceval.components as comp  # type: ignore
        import perceval.algorithm as algo  # type: ignore
        from perceval import catalog  # type: ignore

        return pcvl, comp, catalog, algo
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Hybrid training requires the `perceval` package. "
            "Install it or use the classical mode."
        ) from exc


@dataclass
class HybridConfig:
    num_layers: int = 6
    num_samples: int = 500
    optimizer: Literal["spsa", "cmaes"] = "cmaes"
    sigma: float = 0.3
    hidden_units: int = 10
    quantum_epochs: int = 1
    classical_epochs: int = 50


class HybridTrainer:
    """Encapsulates the hybrid quantum-classical training pipeline."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_classes: int,
        batch_size: int,
        config: HybridConfig,
    ):
        pcvl, comp, catalog, algo = _lazy_import_perceval()
        self.pcvl = pcvl
        self.comp = comp
        self.catalog = catalog
        self.algo = algo

        self.X_train = X_train
        self.y_train = y_train.astype(int)
        self.X_val = X_val
        self.y_val = y_val.astype(int)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.config = config

        self.eff_img_size = X_train.shape[1]
        self.num_modes = self.eff_img_size * 2
        self.num_photons = self.eff_img_size // 2 if self.eff_img_size % 2 == 0 else self.eff_img_size // 2 + 1

        pattern = [1, 0] * self.num_photons + [0] * (self.num_modes - 2 * self.num_photons)
        self.input_state = self.pcvl.BasicState(pattern)

        parity_shift = self.config.num_layers % 2
        indices = [f"{i + parity_shift * self.eff_img_size}" for i in range(self.eff_img_size)]
        condition = f"[{','.join(indices)}] == {self.num_photons}"
        self.post_select = self.pcvl.utils.postselect.PostSelect(condition)

        self.processor = self.pcvl.components.processor.Processor("CliffordClifford2017")
        self._filter_unitary_examples()

    def _to_hamiltonian(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * (x[:, :, 0] + x[:, :, 0].T)

    def _ua(self, x: np.ndarray) -> np.ndarray:
        x_mat = x[:, :, 0]
        x_norm = np.linalg.norm(x_mat, ord=2) or 1.0
        scaled = x_mat / (1.7 * x_norm)
        identity = np.eye(self.eff_img_size)
        block = np.block(
            [
                [scaled, linalg.sqrtm(identity - scaled @ scaled.T)],
                [linalg.sqrtm(identity - scaled.T @ scaled), -scaled.T],
            ]
        )
        return block

    def _is_unitary(self, matrix: np.ndarray) -> bool:
        identity = np.eye(matrix.shape[0])
        return np.allclose(matrix.conj().T @ matrix, identity)

    def _filter_unitary_examples(self) -> None:
        mask_train = [self._is_unitary(self._ua(x)) for x in self.X_train]
        mask_val = [self._is_unitary(self._ua(x)) for x in self.X_val]
        removed_train = len(mask_train) - sum(mask_train)
        removed_val = len(mask_val) - sum(mask_val)
        self.X_train = self.X_train[mask_train]
        self.y_train = self.y_train[mask_train]
        self.X_val = self.X_val[mask_val]
        self.y_val = self.y_val[mask_val]
        if removed_train or removed_val:
            LOG.info(
                "Removed %d train / %d val samples that broke unitarity.",
                removed_train,
                removed_val,
            )

    def _brickwork(self, omega: np.ndarray, num_modes: int) -> "pcvl.Circuit":
        circ = self.pcvl.Circuit(num_modes)
        even_modes = np.arange(0, num_modes - 1, 2)
        odd_modes = np.arange(1, num_modes - 1, 2)
        for idx in even_modes:
            params = omega[int(idx)]
            circ.add(
                int(idx),
                self.catalog["generic 2 mode circuit"].build_circuit(
                    theta=params[0], phi_tl=params[1], phi_bl=params[2], phi_tr=params[3]
                ),
            )
        for idx in odd_modes:
            params = omega[int(idx)]
            circ.add(
                int(idx),
                self.catalog["generic 2 mode circuit"].build_circuit(
                    theta=params[0], phi_tl=params[1], phi_bl=params[2], phi_tr=params[3]
                ),
            )
        return circ

    def _brickwork_bs(self, omega: np.ndarray, num_modes: int) -> "pcvl.Circuit":
        circ = self.pcvl.Circuit(num_modes)
        even_modes = np.arange(0, num_modes - 1, 2)
        odd_modes = np.arange(1, num_modes - 1, 2)
        for idx in even_modes:
            circ.add(int(idx), self.comp.BS(omega[int(idx)]))
        for idx in odd_modes:
            circ.add(int(idx), self.comp.BS(omega[int(idx)]))
        return circ

    def _create_circuit(self, x: np.ndarray, bs_params: np.ndarray, omega_params: np.ndarray) -> "pcvl.Circuit":
        circuit = self.pcvl.Circuit(self.num_modes)
        circuit.add(0, self._brickwork_bs(bs_params, self.eff_img_size))
        for layer in range(self.config.num_layers):
            parity = 1 - layer % 2
            circuit.add(0, self.comp.Unitary(U=self.pcvl.Matrix(self._ua(x))))
            circuit.add(parity * self.eff_img_size, self._brickwork(omega_params[layer], self.eff_img_size))
        return circuit

    def _sample_prob(self, x: np.ndarray, bs_params: np.ndarray, omega_params: np.ndarray) -> Dict:
        self.processor.set_circuit(self._create_circuit(x, bs_params, omega_params))
        self.processor.with_input(self.input_state)
        self.processor.set_postselection(self.post_select)
        sampler = self.algo.Sampler(self.processor, max_shots_per_call=self.config.num_samples)
        return sampler.sample_count(self.config.num_samples)

    def _mean_output(self, x: np.ndarray, bs_params: np.ndarray, omega_params: np.ndarray) -> np.ndarray:
        prob_dist = self._sample_prob(x, bs_params, omega_params)
        mean_output = np.zeros(self.eff_img_size)
        for output, count in prob_dist["results"].items():
            for i in range(self.eff_img_size):
                mean_output[i] += output[i + (self.config.num_layers % 2) * self.eff_img_size] * count
        return mean_output / self.config.num_samples

    def _classical_layer(self, hidden_units: int) -> models.Sequential:
        classifier = models.Sequential(
            [
                layers.Dense(hidden_units, activation="relu", input_dim=self.eff_img_size),
                layers.Dense(self.num_classes),
                layers.Softmax(),
            ]
        )
        classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return classifier

    def _minibatch_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bs_params: np.ndarray,
        omega_params: np.ndarray,
        classifier: models.Sequential,
    ) -> float:
        losses = []
        for idx in range(len(X)):
            mean_output = self._mean_output(X[idx], bs_params, omega_params)
            logits = classifier.predict(mean_output.reshape(1, -1), verbose=0)
            y_true = to_categorical(y[idx], num_classes=self.num_classes).reshape(1, -1)
            losses.append(tf.keras.losses.categorical_crossentropy(y_true, logits).numpy().mean())
        return float(np.mean(losses))

    def _model_accuracy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bs_params: np.ndarray,
        omega_params: np.ndarray,
        classifier: models.Sequential,
    ) -> float:
        predictions = []
        for idx in range(len(X)):
            mean_output = self._mean_output(X[idx], bs_params, omega_params)
            pred = classifier.predict(mean_output.reshape(1, -1), verbose=0)[0].argmax()
            predictions.append(pred)
        return float(metrics.accuracy_score(y, predictions))

    def _features(self, X: np.ndarray, bs_params: np.ndarray, omega_params: np.ndarray) -> np.ndarray:
        features = np.zeros((len(X), self.eff_img_size))
        for idx in range(len(X)):
            features[idx] = self._mean_output(X[idx], bs_params, omega_params)
        return features

    def train(self) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng()
        omega_params = rng.uniform(0, 2 * np.pi, (self.config.num_layers, self.eff_img_size - 1, 4))
        bs_params = rng.uniform(0, 2 * np.pi, self.eff_img_size - 1)
        LOG.info(
            "Initial optical parameters: omega shape %s, BS length %d",
            omega_params.shape,
            len(bs_params),
        )

        if self.config.optimizer == "spsa":
            return self._train_spsa(bs_params, omega_params)
        if self.config.optimizer == "cmaes":
            return self._train_cmaes(bs_params, omega_params)
        raise ValueError(f"Unknown optimizer '{self.config.optimizer}'")

    def _train_spsa(self, bs_params: np.ndarray, omega_params: np.ndarray) -> Dict[str, np.ndarray]:
        classifier = self._classical_layer(self.config.hidden_units)
        omega_shape = omega_params.shape
        bs_len = len(bs_params)
        params = np.concatenate([bs_params, omega_params.flatten()])
        best_params = params.copy()
        best_val_acc = 0.0
        loss_history: list[float] = []
        total_loss_history: list[float] = []
        val_history: list[float] = []

        LOG.info("Training with SPSA for %d epochs", self.config.quantum_epochs)

        for epoch in range(self.config.quantum_epochs):
            LOG.info("Epoch %d/%d", epoch + 1, self.config.quantum_epochs)
            shuffled = np.random.permutation(len(self.X_train))
            num_batches = max(1, len(self.X_train) // self.batch_size)

            for batch_idx in range(num_batches):
                start = batch_idx * self.batch_size
                end = min((batch_idx + 1) * self.batch_size, len(self.X_train))
                batch_indices = shuffled[start:end]
                batch_X = self.X_train[batch_indices]
                batch_y = self.y_train[batch_indices]

                classifier.fit(
                    self._features(batch_X, bs_params, omega_params),
                    to_categorical(batch_y, num_classes=self.num_classes),
                    epochs=self.config.classical_epochs,
                    verbose=0,
                )

                a_k = 0.01 / (batch_idx + 1) ** 0.602
                c_k = 0.05 / (batch_idx + 1) ** 0.101
                delta = 2 * np.random.randint(0, 2, params.shape) - 1

                params_plus = params + c_k * delta
                params_minus = params - c_k * delta

                bs_plus, omega_plus = np.split(params_plus, [bs_len])
                omega_plus = omega_plus.reshape(omega_shape)
                loss_plus = self._minibatch_loss(batch_X, batch_y, bs_plus, omega_plus, classifier)

                bs_minus, omega_minus = np.split(params_minus, [bs_len])
                omega_minus = omega_minus.reshape(omega_shape)
                loss_minus = self._minibatch_loss(batch_X, batch_y, bs_minus, omega_minus, classifier)

                gradient = (loss_plus - loss_minus) / (2 * c_k * delta)
                params = params - a_k * gradient
                loss_history.append((loss_plus + loss_minus) / 2)

            bs_params, omega_params = np.split(params, [bs_len])
            omega_params = omega_params.reshape(omega_shape)

            classifier.fit(
                self._features(self.X_train, bs_params, omega_params),
                to_categorical(self.y_train, num_classes=self.num_classes),
                epochs=self.config.classical_epochs,
                verbose=0,
            )

            epoch_loss = self._minibatch_loss(self.X_train, self.y_train, bs_params, omega_params, classifier)
            epoch_acc = self._model_accuracy(self.X_val, self.y_val, bs_params, omega_params, classifier)
            total_loss_history.append(epoch_loss)
            val_history.append(epoch_acc)
            LOG.info("Epoch loss %.4f | val accuracy %.4f", epoch_loss, epoch_acc)

            if epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_params = params.copy()

        bs_params, omega_params = np.split(best_params, [bs_len])
        omega_params = omega_params.reshape(omega_shape)
        cm = self._evaluate_confusion(bs_params, omega_params, classifier)

        return {
            "BS_params": bs_params,
            "omega_params": omega_params,
            "loss_history": np.array(loss_history),
            "epoch_loss": np.array(total_loss_history),
            "val_accuracy": np.array(val_history),
            "confusion_matrix": cm,
        }

    def _train_cmaes(self, bs_params: np.ndarray, omega_params: np.ndarray) -> Dict[str, np.ndarray]:
        import cma  # type: ignore

        classifier = self._classical_layer(self.config.hidden_units)
        omega_shape = omega_params.shape
        bs_len = len(bs_params)
        params = np.concatenate([bs_params, omega_params.flatten()])

        loss_history: list[float] = []
        val_history: list[float] = []

        LOG.info("Training with CMA-ES for %d epochs", self.config.quantum_epochs)

        for epoch in range(self.config.quantum_epochs):
            LOG.info("Epoch %d/%d", epoch + 1, self.config.quantum_epochs)
            shuffled = np.random.permutation(len(self.X_train))
            num_batches = max(1, len(self.X_train) // self.batch_size)

            for batch_idx in range(num_batches):
                start = batch_idx * self.batch_size
                end = min((batch_idx + 1) * self.batch_size, len(self.X_train))
                batch_indices = shuffled[start:end]
                batch_X = self.X_train[batch_indices]
                batch_y = self.y_train[batch_indices]

                classifier.fit(
                    self._features(batch_X, bs_params, omega_params),
                    to_categorical(batch_y, num_classes=self.num_classes),
                    epochs=self.config.classical_epochs,
                    verbose=0,
                )

                es = cma.CMAEvolutionStrategy(params, self.config.sigma)

                def loss_function(flat_params: np.ndarray) -> float:
                    bs_local, omega_local = np.split(flat_params, [bs_len])
                    omega_local = omega_local.reshape(omega_shape)
                    return self._minibatch_loss(batch_X, batch_y, bs_local, omega_local, classifier)

                es.optimize(loss_function, iterations=1, verb_disp=0)
                params = es.result.xbest
                loss_history.append(es.result.fbest)

            bs_params, omega_params = np.split(params, [bs_len])
            omega_params = omega_params.reshape(omega_shape)

            classifier.fit(
                self._features(self.X_train, bs_params, omega_params),
                to_categorical(self.y_train, num_classes=self.num_classes),
                epochs=self.config.classical_epochs * 2,
                verbose=0,
            )

            val_acc = self._model_accuracy(self.X_val, self.y_val, bs_params, omega_params, classifier)
            val_history.append(val_acc)
            LOG.info("Validation accuracy after epoch %d: %.4f", epoch + 1, val_acc)

        cm = self._evaluate_confusion(bs_params, omega_params, classifier)
        return {
            "BS_params": bs_params,
            "omega_params": omega_params,
            "loss_history": np.array(loss_history),
            "val_accuracy": np.array(val_history),
            "confusion_matrix": cm,
        }

    def _evaluate_confusion(
        self,
        bs_params: np.ndarray,
        omega_params: np.ndarray,
        classifier: models.Sequential,
    ) -> np.ndarray:
        predictions = []
        for idx in range(len(self.X_val)):
            mean_output = self._mean_output(self.X_val[idx], bs_params, omega_params)
            prediction = classifier.predict(mean_output.reshape(1, -1), verbose=0)[0].argmax()
            predictions.append(prediction)
        return metrics.confusion_matrix(self.y_val, predictions)


def train_hybrid(
    data_source: Literal["merlin", "csv"],
    data_dir: Optional[Path],
    output_dir: Path,
    img_size: int,
    num_classes: int,
    pool_size: int,
    batch_size: int,
    config: HybridConfig,
) -> Dict[str, np.ndarray]:
    X_train, y_train, X_val, y_val, _ = load_dataset(
        data_source=data_source,
        data_dir=data_dir,
        img_size=img_size,
        num_classes=num_classes,
        pool_size=pool_size,
        one_hot=False,
    )
    LOG.info("Hybrid dataset shapes train %s | val %s", X_train.shape, X_val.shape)
    trainer = HybridTrainer(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_classes=num_classes,
        batch_size=batch_size,
        config=config,
    )
    results = trainer.train()

    np.save(output_dir / "BS_params.npy", results["BS_params"])
    np.save(output_dir / "omega_params.npy", results["omega_params"])
    np.save(output_dir / "loss_history.npy", results["loss_history"])
    np.save(output_dir / "val_accuracy.npy", results["val_accuracy"])
    np.save(output_dir / "confusion_matrix.npy", results["confusion_matrix"])
    LOG.info("Saved hybrid artefacts to %s", output_dir)

    plot_confusion_matrix(
        results["confusion_matrix"],
        classes=[str(i) for i in range(num_classes)],
        output_path=output_dir / "hybrid_confusion.png",
    )
    return results


# --------------------------------------------------------------------------- #
# CLI plumbing
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Lancelot models.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data-source",
        choices=["merlin", "csv"],
        default="merlin",
        help="Select 'merlin' to use the built-in MerLin MNIST splits or 'csv' to load custom CSV files.",
    )
    common.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing train.csv and val.csv (required when --data-source csv).",
    )
    common.add_argument("--output-dir", type=Path, default=Path("modelparams"), help="Directory to save artefacts")
    common.add_argument("--img-size", type=int, default=28)
    common.add_argument("--num-classes", type=int, default=10)
    common.add_argument("--pool-size", type=int, default=2, help="MaxPool factor applied before training")
    common.add_argument("--batch-size", type=int, default=64)

    classical = subparsers.add_parser("classical", parents=[common], help="Train the classical CNN")
    classical.add_argument("--epochs", type=int, default=30)
    classical.add_argument("--variant", choices=["full", "light"], default="full")
    classical.add_argument("--skip-plots", action="store_true")

    hybrid = subparsers.add_parser("hybrid", parents=[common], help="Train the hybrid photonic model")
    hybrid.add_argument("--quantum-epochs", type=int, default=1)
    hybrid.add_argument("--sigma", type=float, default=0.3)
    hybrid.add_argument("--hidden-units", type=int, default=10)
    hybrid.add_argument("--optimizer", choices=["spsa", "cmaes"], default="cmaes")
    hybrid.add_argument("--num-layers", type=int, default=6)
    hybrid.add_argument("--num-samples", type=int, default=500)

    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    data_dir = args.data_dir.resolve() if args.data_dir else None
    if args.data_source == "csv" and data_dir is None:
        parser.error("--data-dir is required when --data-source csv is selected.")

    output_dir = ensure_output_dir(args.output_dir.resolve())

    if args.mode == "classical":
        train_classical(
            data_source=args.data_source,
            data_dir=data_dir,
            output_dir=output_dir,
            img_size=args.img_size,
            num_classes=args.num_classes,
            pool_size=args.pool_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            variant=args.variant,
            skip_plots=args.skip_plots,
        )
    elif args.mode == "hybrid":
        config = HybridConfig(
            num_layers=args.num_layers,
            num_samples=args.num_samples,
            optimizer=args.optimizer,
            sigma=args.sigma,
            hidden_units=args.hidden_units,
            quantum_epochs=args.quantum_epochs,
        )
        train_hybrid(
            data_source=args.data_source,
            data_dir=data_dir,
            output_dir=output_dir,
            img_size=args.img_size,
            num_classes=args.num_classes,
            pool_size=args.pool_size,
            batch_size=args.batch_size,
            config=config,
        )
    else:
        parser.error(f"Unsupported mode '{args.mode}'")


if __name__ == "__main__":
    main()
