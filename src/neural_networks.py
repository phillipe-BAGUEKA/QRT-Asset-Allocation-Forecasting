"""
Neural-network models for the QRT Asset Allocation Forecasting project.

The module provides three PyTorch binary classifiers:

- ``MLPBinaryClassifier``: dense neural-network baseline;
- ``CNN1DBinaryClassifier``: local-pattern model for ordered return lags;
- ``LSTMBinaryClassifier``: recurrent model for ordered return lags.

All public estimators follow the Scikit-Learn API and implement:

- ``fit``;
- ``predict``;
- ``predict_proba``;
- ``get_params`` / ``set_params`` through ``BaseEstimator``.

They can therefore be cloned and evaluated with the existing
``evaluate_model_on_folds`` and ``diagnostic_training`` functions.

Preprocessing is learned strictly inside ``fit``. Missing-value imputation
and standardisation never use the external validation fold.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RANDOM_STATE = 42
ModelType = Literal["mlp", "cnn1d", "lstm"]
DeviceType = Literal["auto", "cpu", "cuda"]


def set_global_seed(random_state: int = RANDOM_STATE) -> None:
    """
    Set reproducibility seeds for NumPy and PyTorch.

    Parameters
    ----------
    random_state : int, default=42
        Seed used by NumPy, PyTorch CPU and available CUDA devices.
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class BinaryMLP(nn.Module):
    """
    Dense neural network for binary classification.

    Parameters
    ----------
    input_dim : int
        Number of numerical input features.

    hidden_dims : sequence of int
        Number of neurons in each hidden layer.

    dropout : float
        Dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one logit per observation."""
        return self.network(x).squeeze(1)


class BinaryCNN1D(nn.Module):
    """
    One-dimensional convolutional network for ordered return lags.

    Each observation is interpreted as a one-channel sequence. With features
    ordered chronologically, convolutions detect local motifs such as short
    momentum, reversals or alternating return signs.

    Parameters
    ----------
    sequence_length : int
        Number of lags in the input sequence.

    conv_channels : sequence of int
        Number of filters in successive convolutional blocks.

    kernel_size : int
        Width of the temporal convolution kernels.

    dense_dim : int
        Number of neurons in the dense layer after temporal pooling.

    dropout : float
        Dropout probability used in convolutional and dense blocks.
    """

    def __init__(
        self,
        sequence_length: int,
        conv_channels: Sequence[int] = (16, 32),
        kernel_size: int = 3,
        dense_dim: int = 32,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        if sequence_length <= 0:
            raise ValueError("sequence_length must be strictly positive.")

        convolution_layers: list[nn.Module] = []
        input_channels = 1
        padding = kernel_size // 2

        for output_channels in conv_channels:
            convolution_layers.extend(
                [
                    nn.Conv1d(
                        in_channels=input_channels,
                        out_channels=output_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(output_channels),
                    nn.Dropout(dropout),
                ]
            )
            input_channels = output_channels

        self.feature_extractor = nn.Sequential(*convolution_layers)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one logit per observation."""
        x = x.unsqueeze(1)
        x = self.feature_extractor(x)
        x = self.pooling(x)
        return self.classifier(x).squeeze(1)


class BinaryLSTM(nn.Module):
    """
    LSTM network for ordered return lags.

    Parameters
    ----------
    hidden_size : int
        Number of hidden units in the recurrent state.

    num_layers : int
        Number of stacked LSTM layers.

    dense_dim : int
        Number of neurons in the dense layer after the LSTM.

    dropout : float
        Dropout probability. Recurrent dropout is active only when
        ``num_layers`` is greater than one.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 1,
        dense_dim: int = 32,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        recurrent_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one logit per observation."""
        x = x.unsqueeze(-1)
        _, (hidden_state, _) = self.lstm(x)
        last_hidden_state = hidden_state[-1]
        return self.classifier(last_hidden_state).squeeze(1)


class TorchBinaryClassifier(ClassifierMixin, BaseEstimator):
    """
    Scikit-Learn compatible PyTorch binary classifier.

    The estimator performs a chronological internal train/validation split
    inside the external training fold. Early stopping selects the number of
    epochs, after which the network is reinitialised and trained on the
    complete external training fold for the selected number of epochs.

    Parameters
    ----------
    model_type : {"mlp", "cnn1d", "lstm"}
        Neural architecture to construct.

    hidden_dims : tuple of int
        Hidden-layer dimensions used by the MLP.

    conv_channels : tuple of int
        Convolution filter counts used by the CNN.

    kernel_size : int
        Temporal kernel width used by the CNN.

    dense_dim : int
        Dense hidden dimension used by the CNN and LSTM.

    lstm_hidden_size : int
        Hidden-state dimension used by the LSTM.

    lstm_num_layers : int
        Number of stacked LSTM layers.

    dropout : float
        Dropout probability.

    learning_rate : float
        AdamW learning rate.

    weight_decay : float
        AdamW L2 regularisation coefficient.

    batch_size : int
        Mini-batch size.

    max_epochs : int
        Maximum number of epochs used during early stopping.

    patience : int
        Number of non-improving validation epochs tolerated.

    min_delta : float
        Minimum validation-loss reduction considered an improvement.

    validation_fraction : float
        Fraction of the external training fold reserved chronologically for
        internal early stopping.

    threshold : float
        Probability threshold used by ``predict``.

    reverse_sequence : bool
        Reverse feature order before CNN/LSTM training. Keep ``True`` when
        the input columns are ordered ``RET_1, ..., RET_20`` and ``RET_1``
        is the most recent lag.

    random_state : int
        Reproducibility seed.

    device : {"auto", "cpu", "cuda"}
        Training device. ``"auto"`` uses CUDA when available.

    verbose : bool
        Whether to print epoch-level early-stopping information.
    """

    def __init__(
        self,
        model_type: ModelType = "mlp",
        hidden_dims: tuple[int, ...] = (64, 32),
        conv_channels: tuple[int, ...] = (16, 32),
        kernel_size: int = 3,
        dense_dim: int = 32,
        lstm_hidden_size: int = 32,
        lstm_num_layers: int = 1,
        dropout: float = 0.20,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 1024,
        max_epochs: int = 50,
        patience: int = 5,
        min_delta: float = 1e-4,
        validation_fraction: float = 0.15,
        threshold: float = 0.50,
        reverse_sequence: bool = True,
        random_state: int = RANDOM_STATE,
        device: DeviceType = "auto",
        verbose: bool = False,
    ) -> None:
        self.model_type = model_type
        self.hidden_dims = hidden_dims
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.dense_dim = dense_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.validation_fraction = validation_fraction
        self.threshold = threshold
        self.reverse_sequence = reverse_sequence
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _validate_hyperparameters(self) -> None:
        """Validate estimator configuration."""
        if self.model_type not in {"mlp", "cnn1d", "lstm"}:
            raise ValueError(
                "model_type must be 'mlp', 'cnn1d' or 'lstm'."
            )

        if not self.hidden_dims or any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError(
                "hidden_dims must contain strictly positive integers."
            )

        if not self.conv_channels or any(
            channel <= 0 for channel in self.conv_channels
        ):
            raise ValueError(
                "conv_channels must contain strictly positive integers."
            )

        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be strictly positive.")

        if self.dense_dim <= 0:
            raise ValueError("dense_dim must be strictly positive.")

        if self.lstm_hidden_size <= 0:
            raise ValueError(
                "lstm_hidden_size must be strictly positive."
            )

        if self.lstm_num_layers <= 0:
            raise ValueError(
                "lstm_num_layers must be strictly positive."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be strictly positive."
            )

        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be strictly positive.")

        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be strictly positive.")

        if self.patience <= 0:
            raise ValueError("patience must be strictly positive.")

        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative.")

        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError(
                "validation_fraction must lie in (0, 0.5)."
            )

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1].")

        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError(
                "device must be 'auto', 'cpu' or 'cuda'."
            )

    def _resolve_device(self) -> torch.device:
        """Resolve the requested PyTorch device."""
        if self.device == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return torch.device(self.device)

    def _validate_X(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        fitting: bool,
    ) -> np.ndarray:
        """Validate and convert the feature matrix."""
        if isinstance(X, pd.DataFrame):
            if fitting:
                self.feature_names_in_ = np.asarray(
                    X.columns,
                    dtype=object,
                )
            elif hasattr(self, "feature_names_in_"):
                observed_columns = np.asarray(X.columns, dtype=object)

                if not np.array_equal(
                    observed_columns,
                    self.feature_names_in_,
                ):
                    raise ValueError(
                        "Prediction features differ from training features "
                        "or are not in the same order."
                    )

            X_array = X.to_numpy(dtype=np.float64, copy=True)

        else:
            X_array = np.asarray(X, dtype=np.float64)

        if X_array.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix.")

        if X_array.shape[0] == 0:
            raise ValueError("X contains no observation.")

        if X_array.shape[1] == 0:
            raise ValueError("X contains no feature.")

        if fitting:
            self.n_features_in_ = X_array.shape[1]
        elif X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                "Prediction feature count differs from training."
            )

        X_array[np.isinf(X_array)] = np.nan
        return X_array

    @staticmethod
    def _validate_y(y: pd.Series | np.ndarray) -> np.ndarray:
        """Validate and convert the binary target."""
        y_array = np.asarray(y).reshape(-1)

        if y_array.size == 0:
            raise ValueError("y contains no observation.")

        if pd.isna(y_array).any():
            raise ValueError("y contains missing values.")

        observed_classes = set(np.unique(y_array).tolist())

        if not observed_classes.issubset({0, 1}):
            raise ValueError(
                f"y must be binary. Found: {sorted(observed_classes)}."
            )

        if len(observed_classes) != 2:
            raise ValueError(
                "Training y must contain both binary classes."
            )

        return y_array.astype(np.float32)

    def _build_network(self, input_dim: int) -> nn.Module:
        """Build the requested neural architecture."""
        if self.model_type == "mlp":
            return BinaryMLP(
                input_dim=input_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
            )

        if self.model_type == "cnn1d":
            return BinaryCNN1D(
                sequence_length=input_dim,
                conv_channels=self.conv_channels,
                kernel_size=self.kernel_size,
                dense_dim=self.dense_dim,
                dropout=self.dropout,
            )

        return BinaryLSTM(
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            dense_dim=self.dense_dim,
            dropout=self.dropout,
        )

    def _prepare_sequence_order(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Reverse lag order for sequence models when requested."""
        if (
            self.model_type in {"cnn1d", "lstm"}
            and self.reverse_sequence
        ):
            return X[:, ::-1].copy()

        return X

    def _create_loader(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        *,
        shuffle: bool,
    ) -> DataLoader:
        """Create a PyTorch DataLoader."""
        X_tensor = torch.from_numpy(
            X.astype(np.float32, copy=False)
        )

        if y is None:
            dataset = TensorDataset(X_tensor)
        else:
            y_tensor = torch.from_numpy(
                y.astype(np.float32, copy=False)
            )
            dataset = TensorDataset(X_tensor, y_tensor)

        generator = torch.Generator()
        generator.manual_seed(self.random_state)

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            generator=generator if shuffle else None,
        )

    def _train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_function: nn.Module,
    ) -> float:
        """Train a network for one epoch and return mean loss."""
        model.train()
        total_loss = 0.0
        total_observations = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device_)
            y_batch = y_batch.to(self.device_)

            optimizer.zero_grad(set_to_none=True)

            logits = model(X_batch)
            loss = loss_function(logits, y_batch)

            loss.backward()
            optimizer.step()

            batch_size = X_batch.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_observations += batch_size

        return total_loss / total_observations

    def _evaluate_loss(
        self,
        model: nn.Module,
        loader: DataLoader,
        loss_function: nn.Module,
    ) -> float:
        """Evaluate mean binary cross-entropy loss."""
        model.eval()
        total_loss = 0.0
        total_observations = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device_)
                y_batch = y_batch.to(self.device_)

                logits = model(X_batch)
                loss = loss_function(logits, y_batch)

                batch_size = X_batch.shape[0]
                total_loss += float(loss.item()) * batch_size
                total_observations += batch_size

        return total_loss / total_observations

    def _select_epoch_count(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
    ) -> tuple[int, list[dict[str, float]]]:
        """Select the epoch count with chronological early stopping."""
        set_global_seed(self.random_state)

        model = self._build_network(
            input_dim=X_train.shape[1]
        ).to(self.device_)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        loss_function = nn.BCEWithLogitsLoss()

        train_loader = self._create_loader(
            X_train,
            y_train,
            shuffle=True,
        )

        valid_loader = self._create_loader(
            X_valid,
            y_valid,
            shuffle=False,
        )

        best_epoch = 1
        best_loss = np.inf
        best_state = deepcopy(model.state_dict())
        epochs_without_improvement = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_function=loss_function,
            )

            valid_loss = self._evaluate_loss(
                model=model,
                loader=valid_loader,
                loss_function=loss_function,
            )

            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                }
            )

            if self.verbose:
                print(
                    f"Epoch {epoch:03d} | "
                    f"train_loss={train_loss:.6f} | "
                    f"valid_loss={valid_loss:.6f}"
                )

            if valid_loss < best_loss - self.min_delta:
                best_loss = valid_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                break

        model.load_state_dict(best_state)
        return best_epoch, history

    def _fit_final_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int,
    ) -> nn.Module:
        """Train a fresh network on the complete external training fold."""
        set_global_seed(self.random_state)

        model = self._build_network(
            input_dim=X.shape[1]
        ).to(self.device_)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        loss_function = nn.BCEWithLogitsLoss()

        train_loader = self._create_loader(
            X,
            y,
            shuffle=True,
        )

        for epoch in range(1, n_epochs + 1):
            train_loss = self._train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_function=loss_function,
            )

            if self.verbose:
                print(
                    f"Final training epoch {epoch:03d}/{n_epochs:03d} | "
                    f"loss={train_loss:.6f}"
                )

        return model

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "TorchBinaryClassifier":
        """
        Fit the neural classifier.

        The final part of the supplied training data is used only to choose
        the epoch count. The network is then reinitialised and trained on all
        supplied observations for the selected number of epochs.
        """
        self._validate_hyperparameters()
        set_global_seed(self.random_state)

        X_array = self._validate_X(X, fitting=True)
        y_array = self._validate_y(y)

        if len(X_array) != len(y_array):
            raise ValueError(
                "X and y do not contain the same number of observations."
            )

        if len(X_array) < 20:
            raise ValueError(
                "At least 20 observations are required for training."
            )

        self.device_ = self._resolve_device()
        self.classes_ = np.asarray([0, 1])

        n_valid = max(
            1,
            int(np.ceil(
                len(X_array) * self.validation_fraction
            )),
        )
        n_train = len(X_array) - n_valid

        if n_train < 2:
            raise ValueError(
                "The internal training subset is too small."
            )

        X_subtrain_raw = X_array[:n_train]
        X_internal_valid_raw = X_array[n_train:]

        y_subtrain = y_array[:n_train]
        y_internal_valid = y_array[n_train:]

        if len(np.unique(y_subtrain)) != 2:
            raise ValueError(
                "The internal training subset must contain both classes."
            )

        self.selection_imputer_ = SimpleImputer(
            strategy="median"
        )
        self.selection_scaler_ = StandardScaler()

        X_subtrain = self.selection_imputer_.fit_transform(
            X_subtrain_raw
        )
        X_internal_valid = self.selection_imputer_.transform(
            X_internal_valid_raw
        )

        X_subtrain = self.selection_scaler_.fit_transform(
            X_subtrain
        )
        X_internal_valid = self.selection_scaler_.transform(
            X_internal_valid
        )

        X_subtrain = self._prepare_sequence_order(X_subtrain)
        X_internal_valid = self._prepare_sequence_order(
            X_internal_valid
        )

        self.best_epoch_, self.training_history_ = (
            self._select_epoch_count(
                X_train=X_subtrain,
                y_train=y_subtrain,
                X_valid=X_internal_valid,
                y_valid=y_internal_valid,
            )
        )

        self.imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()

        X_full = self.imputer_.fit_transform(X_array)
        X_full = self.scaler_.fit_transform(X_full)
        X_full = self._prepare_sequence_order(X_full)

        self.model_ = self._fit_final_model(
            X=X_full,
            y=y_array,
            n_epochs=self.best_epoch_,
        )

        self.model_.eval()
        return self

    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """
        Return probabilities for classes 0 and 1.
        """
        check_is_fitted(
            self,
            attributes=[
                "model_",
                "imputer_",
                "scaler_",
                "classes_",
                "best_epoch_",
            ],
        )

        X_array = self._validate_X(X, fitting=False)
        X_array = self.imputer_.transform(X_array)
        X_array = self.scaler_.transform(X_array)
        X_array = self._prepare_sequence_order(X_array)

        loader = self._create_loader(
            X_array,
            y=None,
            shuffle=False,
        )

        positive_probabilities: list[np.ndarray] = []

        self.model_.eval()

        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device_)
                logits = self.model_(X_batch)
                probabilities = torch.sigmoid(logits)

                positive_probabilities.append(
                    probabilities.cpu().numpy()
                )

        positive_proba = np.concatenate(
            positive_probabilities
        ).astype(np.float64)

        return np.column_stack(
            [
                1.0 - positive_proba,
                positive_proba,
            ]
        )

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """
        Return binary predictions using the configured threshold.
        """
        positive_proba = self.predict_proba(X)[:, 1]
        return (positive_proba >= self.threshold).astype(int)


def build_mlp_classifier(
    hidden_dims: tuple[int, ...] = (64, 32),
    dropout: float = 0.20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 1024,
    max_epochs: int = 50,
    patience: int = 5,
    validation_fraction: float = 0.15,
    random_state: int = RANDOM_STATE,
    device: DeviceType = "auto",
    verbose: bool = False,
) -> TorchBinaryClassifier:
    """
    Build the MLP baseline.

    Returns
    -------
    TorchBinaryClassifier
        Untrained Scikit-Learn compatible estimator.
    """
    return TorchBinaryClassifier(
        model_type="mlp",
        hidden_dims=hidden_dims,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        validation_fraction=validation_fraction,
        reverse_sequence=False,
        random_state=random_state,
        device=device,
        verbose=verbose,
    )


def build_cnn1d_classifier(
    conv_channels: tuple[int, ...] = (16, 32),
    kernel_size: int = 3,
    dense_dim: int = 32,
    dropout: float = 0.20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 1024,
    max_epochs: int = 50,
    patience: int = 5,
    validation_fraction: float = 0.15,
    reverse_sequence: bool = True,
    random_state: int = RANDOM_STATE,
    device: DeviceType = "auto",
    verbose: bool = False,
) -> TorchBinaryClassifier:
    """
    Build the one-dimensional CNN challenger.

    Returns
    -------
    TorchBinaryClassifier
        Untrained Scikit-Learn compatible estimator.
    """
    return TorchBinaryClassifier(
        model_type="cnn1d",
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        dense_dim=dense_dim,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        validation_fraction=validation_fraction,
        reverse_sequence=reverse_sequence,
        random_state=random_state,
        device=device,
        verbose=verbose,
    )


def build_lstm_classifier(
    hidden_size: int = 32,
    num_layers: int = 1,
    dense_dim: int = 32,
    dropout: float = 0.20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 1024,
    max_epochs: int = 50,
    patience: int = 5,
    validation_fraction: float = 0.15,
    reverse_sequence: bool = True,
    random_state: int = RANDOM_STATE,
    device: DeviceType = "auto",
    verbose: bool = False,
) -> TorchBinaryClassifier:
    """
    Build the LSTM experimental model.

    Returns
    -------
    TorchBinaryClassifier
        Untrained Scikit-Learn compatible estimator.
    """
    return TorchBinaryClassifier(
        model_type="lstm",
        dense_dim=dense_dim,
        lstm_hidden_size=hidden_size,
        lstm_num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        validation_fraction=validation_fraction,
        reverse_sequence=reverse_sequence,
        random_state=random_state,
        device=device,
        verbose=verbose,
    )
