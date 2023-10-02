import numpy as np
from scipy.special import softmax
from seaborn import scatterplot
from matplotlib.pyplot import imshow, plot, colorbar
from pandas import get_dummies


class LogRegressorClassifier:
    @staticmethod
    def _sigmoide(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _multidim_gradient_loss(X: np.ndarray, theta: np.ndarray, p: np.ndarray):
        return (1 / X.shape[0]) * (X.T @ (softmax(X @ theta, axis=1) - p))

    @staticmethod
    def _binary_gradient_loss(X: np.ndarray, theta: np.ndarray, p: np.ndarray):
        return (
            (1 / X.shape[0]) * X.T @ (LogRegressorClassifier._sigmoide(X @ theta) - p)
        )

    @staticmethod
    def _get_binary_start_coef(X: np.ndarray):
        return np.random.normal(size=(X.shape[1]))

    @staticmethod
    def _get_multidim_start_coef(X: np.ndarray):
        return np.random.normal(size=(X.shape[1], 3))

    @staticmethod
    def _prepare_x_for_intercept(X):
        return np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)  # type:ignore

    @staticmethod
    def _prepare_p_for_multidim(p):
        return get_dummies(p).to_numpy()

    @staticmethod
    def _prepare_p_for_binary(p):
        return np.squeeze(1 - get_dummies(p, drop_first=True).to_numpy())

    def _multidim_train(self):
        if self.X is None or self.p is None:
            raise ValueError("X must be a ndarray")
        theta = self._get_multidim_start_coef(self.X)  # type: ignore
        for _ in range(self.max_iter):
            theta = theta - self.alpha * self._multidim_gradient_loss(self.X, theta, self.p)  # type: ignore
        return theta

    def _binary_train(self):
        if self.X is None or self.p is None:
            raise ValueError("X must be a ndarray")
        theta = self._get_binary_start_coef(self.X)  # type: ignore
        for _ in range(self.max_iter):
            theta = theta - self.alpha * self._binary_gradient_loss(
                self.X, theta, self.p  # type: ignore
            )
        return theta

    def _get_plot_grid(self):
        x_length = np.max(self.X[:, 0]) - np.min(self.X[:, 0])  # type: ignore
        y_length = np.max(self.X[:, 1]) - np.min(self.X[:, 1])  # type: ignore
        x_bornes = (np.min(self.X[:, 0]) - 0.05 * x_length, np.max(self.X[:, 0]) + 0.05 * x_length)  # type: ignore
        y_bornes = (np.min(self.X[:, 1]) - 0.05 * y_length, np.max(self.X[:, 1]) + 0.05 * y_length)  # type: ignore
        x_axis = np.linspace(*x_bornes, num=500)  # type: ignore
        y_axis = np.linspace(*y_bornes, num=500)  # type: ignore
        grid_x, grid_y = np.meshgrid(x_axis, y_axis)
        return grid_x, grid_y, x_bornes, y_bornes

    def _plot_proba_binary2D(self):
        grid_x, grid_y, x_bornes, y_bornes = self._get_plot_grid()
        proba = lambda x, y: self._sigmoide(
            x * self.coef_[0] + y * self.coef_[1] + (self.coef_[2] if self.intercept else 0)  # type: ignore
        )
        imshow(proba(grid_x, grid_y), origin="lower", extent=(*x_bornes, *y_bornes))  # type: ignore
        scatterplot(x=self.X[:, 0], y=self.X[:, 1], hue=self.p)  # type: ignore
        colorbar()

    def _plot_proba_multidim2D(self):
        logits = np.zeros((500, 500, self.coef_.shape[1]))  # type: ignore
        grid_x, grid_y, x_bornes, y_bornes = self._get_plot_grid()
        for i in range(self.coef_.shape[1]):  # type: ignore
            logits[:, :, i] = grid_x * self.coef_[0, i] + grid_y * self.coef_[1, i] + (self.coef_[2, i] if self.intercept else 0)  # type: ignore
        image = softmax(logits, axis=self.coef_.shape[1] - 1)  # type: ignore
        imshow(image, origin="lower", extent=(*x_bornes, *y_bornes))  # type: ignore
        scatterplot(
            x=self.X[:, 0],  # type: ignore
            y=self.X[:, 1],  # type: ignore
            hue=self.initial_p,
            edgecolor="black",  # type: ignore
            linewidth=1,
            palette=["r", "g", "b"],
        )

    def plot_binary2D(self):
        x_length = np.max(self.X[:, 0]) - np.min(self.X[:, 0])  # type: ignore
        x_bornes = (np.min(self.X[:, 0]) - 0.05 * x_length, np.max(self.X[:, 0]) + 0.05 * x_length)  # type: ignore
        x_axis = np.linspace(*x_bornes, num=300)
        scatterplot(x=self.X[:, 0], y=self.X[:, 1], hue=self.initial_p)  # type: ignore
        sep = (
            self.sep_line_with_intercept(x_axis[:-20])
            if self.intercept
            else self.sep_line(x_axis[:-20])
        )
        plot(x_axis[:-20], sep, color="r")

    def sep_line(self, x):
        return -(self.coef_[0] / self.coef_[1]) * x  # type: ignore

    def sep_line_with_intercept(self, x):
        return (-self.coef_[2] - self.coef_[0] * x) / self.coef_[1]  # type: ignore

    def __init__(
        self, multidim: bool, intercept: bool, max_iter: int = 1000, alpha: float = 1
    ):
        self.multidim = multidim
        self.max_iter = max_iter
        self.intercept = intercept
        self.alpha = alpha
        self.X = (None,)
        self.p = (None,)
        self.initial_p = (None,)
        self.initial_X = (None,)
        self.coef_ = (None,)

    def fit(self, X: np.ndarray, p: np.ndarray):
        self.initial_p = p
        self.initial_X = X
        self.X = (self._prepare_x_for_intercept(X) if self.intercept else X).astype(
            float
        )
        self.p = (
            self._prepare_p_for_multidim(p)
            if self.multidim
            else self._prepare_p_for_binary(p)
        )
        self.coef_ = self._multidim_train() if self.multidim else self._binary_train()
        return self

    def _inference_binary(self, Xtest):
        return Xtest @ self.coef_ > 0  # type: ignore

    def score_binary(self, test_X, test_p):
        test_p = self._prepare_p_for_binary(test_p)
        test_X = self._prepare_x_for_intercept(test_X)
        return (self._inference_binary(test_X) == test_p).sum() / test_p.size  # type: ignore

    def plot_proba_2D(self):
        return (
            self._plot_proba_multidim2D()
            if self.multidim
            else self._plot_proba_binary2D()
        )
