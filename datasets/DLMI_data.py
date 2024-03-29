from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.load_data import load_data, load_X_y
    from benchmark_utils.load_data import load_data_bio


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "DLMI"

    requirements = []

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        csv_path_train = "dataset/dataDLMI-main/trainset/trainset_true.csv"
        images_path_train = "dataset/dataDLMI-main/trainset/"

        data_train = load_data(csv_path_train, images_path_train)

        X_img, y = load_X_y(data_train)

        X_bio, _ = load_data_bio(data_train)

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(X_img=X_img,
                    X_bio=X_bio,
                    y=y)
