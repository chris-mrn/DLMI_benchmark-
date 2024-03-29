from benchopt import BaseObjective, safe_import_context
from sklearn.metrics import balanced_accuracy_score

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.model_selection import train_test_split
    from benchmark_utils.processing import flat_set_for_deep
    from benchmark_utils.processing import flat_set_img_bio


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "DLMI"

    # URL of the main repo for this benchmark.
    url = "https://github.com/#ORG/#BENCHMARK_NAME"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    # This means the OLS objective will have a parameter `self.whiten_y`.
    parameters = {}

    # List of packages needed to run the benchmark.
    # They are installed with conda; to use pip, use 'pip:packagename'. To
    # install from a specific conda channel, use 'channelname:packagename'.
    # Packages that are not necessary to the whole benchmark but only to some
    # solvers or datasets should be declared in Dataset or Solver (see
    # simulated.py and python-gd.py).
    # Example syntax: requirements = ['numpy', 'pip:jax', 'pytorch:pytorch']
    requirements = []

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5"

    def set_data(self, X_img, X_bio, y):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.

        # Split the data into training and testing sets
        X_img_train, X_img_test, y_train, y_test = train_test_split(
                                                    X_img,
                                                    y,
                                                    random_state=42)

        X_bio_train, X_bio_test, _, _ = train_test_split(
                                                    X_bio,
                                                    y,
                                                    random_state=42)

        self.X_img_train = X_img_train
        self.X_img_test = X_img_test

        self.X_bio_train = X_bio_train
        self.X_bio_test = X_bio_test

        self.y_train = y_train
        self.y_test = y_test

    def evaluate_result(self, model, data):
        # The keyword arguments of this function are the keys of the
        # dictionary returned by `Solver.get_result`. This defines the
        # benchmark's API to pass solvers' result. This is customizable for
        # each benchmark.
        y_train = self.y_train
        y_test = self.y_test

        if data == 'bio':
            X_train = self.X_bio_train
            X_test = self.X_bio_test

        if data == 'img':
            X_train, y_train = flat_set_for_deep(self.X_img_train, y_train)
            X_test, y_test = flat_set_for_deep(self.X_img_test, y_test)

        if data == 'img+bio':
            X_train, y_train = flat_set_img_bio(self.X_img_train,
                                                self.X_bio_train,
                                                y_train)
            X_test, y_test = flat_set_img_bio(self.X_img_test,
                                              self.X_bio_test,
                                              y_test)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        score_test = balanced_accuracy_score(y_test, y_pred_test)
        score_train = balanced_accuracy_score(y_train, y_pred_train)

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            score_test=score_test,
            score_train=score_train,
            value=1-score_test
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        return dict(beta=np.zeros(self.X.shape[1]))

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.

        return dict(
            X_img=self.X_img_train,
            X_bio=self.X_bio_train,
            y=self.y_train,
        )
