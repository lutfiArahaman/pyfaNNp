"""Neural surrogate for the outranking.

The network is trained to map a criterion vector to the PROMETHEE net flow
that the full pairwise outranking would assign it. Once fitted it scores new
alternatives in constant time per alternative, where the exact computation is
quadratic in the number of alternatives and must be re-run in full whenever
the set changes.

The surrogate approximates the outranking; it does not replace it. The exact
flows remain available on the result object, and the surrogate's training
score is reported so that the quality of the approximation can be judged.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ANNSurrogate"]


class ANNSurrogate:
    """Feed-forward network wrapping a scikit-learn ``MLPRegressor``.

    Parameters
    ----------
    hidden : tuple of int
        Hidden layer sizes.
    epochs : int
        Maximum training iterations.
    random_state : int, optional
        Seed for weight initialisation, for reproducible results.
    backend : {"sklearn", "torch"}
        Only ``"sklearn"`` is implemented; ``"torch"`` is reserved.
    learning_rate : float
        Initial learning rate for the Adam solver.
    standardize : bool
        Standardise inputs and target before fitting. Recommended: criterion
        values are typically on incomparable scales.

    Attributes set by :meth:`fit`
    -----------------------------
    train_score_ : float
        Coefficient of determination on the training data.
    """

    def __init__(
        self,
        hidden=(32, 16),
        epochs: int = 300,
        random_state=None,
        backend: str = "sklearn",
        learning_rate: float = 1e-3,
        standardize: bool = True,
    ):
        self.hidden = tuple(hidden)
        self.epochs = int(epochs)
        self.random_state = random_state
        self.backend = backend
        self.learning_rate = float(learning_rate)
        self.standardize = standardize

        self._model = None
        self._x_mean = None
        self._x_std = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self.train_score_ = None

    # ------------------------------------------------------------------ API

    def fit(self, X, y) -> ANNSurrogate:
        """Fit the surrogate on criterion vectors ``X`` and net flows ``y``."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D; got shape {X.shape}")
        if X.shape[0] != y.size:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {y.size} entries"
            )

        Xs, ys = self._fit_scalers(X, y)
        self._model = self._build()
        self._model.fit(Xs, ys)

        self.train_score_ = float(self._model.score(Xs, ys))
        return self

    def predict(self, X) -> np.ndarray:
        """Predict net flows for new criterion vectors."""
        if self._model is None:
            raise RuntimeError("the surrogate has not been fitted")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        Xs = (X - self._x_mean) / self._x_std if self.standardize else X
        return self._model.predict(Xs) * self._y_std + self._y_mean

    # ------------------------------------------------------------ internals

    def _build(self):
        if self.backend == "torch":
            raise NotImplementedError(
                "the torch backend is not implemented in this scaffold; "
                "use backend='sklearn'"
            )
        if self.backend != "sklearn":
            raise ValueError(f"unknown backend {self.backend!r}")

        try:
            from sklearn.neural_network import MLPRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "the sklearn backend requires scikit-learn; "
                "install it with `pip install scikit-learn`"
            ) from exc

        return MLPRegressor(
            hidden_layer_sizes=self.hidden,
            max_iter=self.epochs,
            random_state=self.random_state,
            learning_rate_init=self.learning_rate,
            solver="adam",
        )

    def _fit_scalers(self, X, y):
        if not self.standardize:
            self._x_mean, self._x_std = 0.0, 1.0
            self._y_mean, self._y_std = 0.0, 1.0
            return X, y

        self._x_mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0          # constant criteria carry no information
        self._x_std = std

        self._y_mean = float(y.mean())
        y_std = float(y.std())
        self._y_std = y_std if y_std > 0 else 1.0

        return (X - self._x_mean) / self._x_std, (y - self._y_mean) / self._y_std
