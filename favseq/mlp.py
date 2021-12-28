import numpy as np
import sklearn.neural_network as sknn

from scipy.special import xlogy
from scipy.special import expit as logistic_sigmoid

from sklearn.neural_network._multilayer_perceptron import safe_sparse_dot  # noqa

from sklearn.neural_network._multilayer_perceptron import DERIVATIVES  # noqa
from sklearn.neural_network._multilayer_perceptron import LOSS_FUNCTIONS as LOSS_FUNCTIONS_  # noqa


def identity(X):  # noqa
    return X


def logistic(X):  # noqa
    return logistic_sigmoid(X, out=X)


def tanh(X):  # noqa
    return np.tanh(X, out=X)


def relu(X):  # noqa
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X


def softmax(X):  # noqa
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]

    return X


ACTIVATIONS = {
    'identity': identity,
    'tanh': tanh,
    'logistic': logistic,
    'relu': relu,
    'softmax': softmax
}


def weighted_log_loss(y_true, y_prob, weights):
    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    loss = -xlogy(y_true, y_prob)
    weighted_loss = weights * loss
    weighted_scalar_loss = (weighted_loss.sum() / y_prob.shape[0]) / np.sum(weights)

    return weighted_scalar_loss


LOSS_FUNCTIONS = {n: fn for n, fn in LOSS_FUNCTIONS_.items()}
LOSS_FUNCTIONS['weighted_log_loss'] = weighted_log_loss


class MLPClassifierWithGrad(sknn.MLPClassifier):
    """
    CREDITS: https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/neural_network/_multilayer_perceptron.py
    """

    def __init__(self, hidden_layer_sizes=(100,), dropout=(0.0,), activation="relu",
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001,
                 power_t=0.5, max_iter=200, shuffle=True,
                 random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10, max_fun=15000):

        super(MLPClassifierWithGrad, self).__init__(
            hidden_layer_sizes=hidden_layer_sizes, activation=activation,
            solver=solver, alpha=alpha,
            batch_size=batch_size, learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t, max_iter=max_iter, shuffle=shuffle,
            random_state=random_state, tol=tol,
            verbose=verbose, warm_start=warm_start, momentum=momentum,
            nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping,
            validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2,
            epsilon=epsilon, n_iter_no_change=n_iter_no_change, max_fun=max_fun
        )

        self.loss = 'weighted_log_loss'

        self.dropout = dropout
        self._training = True
        self._drop_applied = dict()
        self.class_weights_ = None

    def _update_no_improvement_count(self, early_stopping, X_val, y_val):
        if not hasattr(self, 'validation_curve_'):
            self.validation_curve_ = list()

        if early_stopping:
            y_pred = self.predict(X_val)
            loss = np.mean((y_val - y_pred) ** 2)
            self.validation_curve_.append(loss)

        super()._update_no_improvement_count(early_stopping, X_val, y_val)

    def _accumulate_grad(self, X, y, coef_grads):  # noqa
        if not hasattr(self, '_coef_grads'):
            self._coef_grads = np.abs(coef_grads).sum(axis=0)
        else:
            self._coef_grads += np.abs(coef_grads).sum(axis=0)

    def _predict(self, X):
        self._training = False
        return super(MLPClassifierWithGrad, self)._forward_pass_fast(X)

    def _forward_pass(self, activations):
        """Perform a forward pass on the network by computing the values
        of the neurons in the hidden layers and the output layer.

        Parameters
        ----------
        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.
        """
        hidden_activation = ACTIVATIONS[self.activation]
        # Iterate over the hidden layers

        for i in range(self.n_layers_ - 1):
            activations[i + 1] = safe_sparse_dot(activations[i], self.coefs_[i])
            activations[i + 1] += self.intercepts_[i]

            # For the hidden layers
            if (i + 1) != (self.n_layers_ - 1):
                activations[i + 1] = hidden_activation(activations[i + 1])

                # dropout
                if self._training:
                    p = self.dropout[i]
                    mask = np.random.binomial(1, 1.0 - p, activations[i + 1].shape) / (1.0 - p)
                    activations[i + 1] = activations[i + 1] * mask
                    self._drop_applied[i + 1] = mask

        # For the last layer
        output_activation = ACTIVATIONS[self.out_activation_]
        activations[i + 1] = output_activation(activations[i + 1])  # noqa

        return activations

    def _backprop(self, X, y, activations, deltas, coef_grads, intercept_grads):
        self._training = True
        n_samples = X.shape[0]

        # Forward propagate
        activations = self._forward_pass(activations)

        # Get loss
        loss_func_name = self.loss

        if loss_func_name == 'log_loss' and self.out_activation_ == 'logistic':
            loss_func_name = 'binary_log_loss'

        loss_func_args = [y, activations[-1]]
        if loss_func_name == 'weighted_log_loss':
            loss_func_args.append(self.class_weights_)

        loss = LOSS_FUNCTIONS[loss_func_name](*loss_func_args)

        # Add L2 regularization term to loss
        values = np.sum(np.array([np.dot(s.ravel(), s.ravel()) for s in self.coefs_]))
        loss += (0.5 * self.alpha) * values / n_samples

        # Backward propagate
        last = self.n_layers_ - 2

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        deltas[last] = activations[-1] - y

        # Compute gradient for the last layer
        self._compute_loss_grad(last, n_samples, activations, deltas, coef_grads, intercept_grads)

        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            inplace_derivative = DERIVATIVES[self.activation]
            inplace_derivative(activations[i], deltas[i - 1])

            deltas[i - 1] *= self._drop_applied[i]

            self._compute_loss_grad(i - 1, n_samples, activations, deltas, coef_grads, intercept_grads)

        self._accumulate_grad(X, y, (self.coefs_[0] @ deltas[0].T).T)

        return loss, coef_grads, intercept_grads

    @property
    def feature_importances_(self):
        return self._coef_grads

    def fit(self, X, y):
        labels, counts = np.unique(y, return_counts=True)
        self.class_weights_ = np.ones(len(labels))
        self.class_weights_[labels] = 1.0 / counts

        return super(MLPClassifierWithGrad, self).fit(X, y)
