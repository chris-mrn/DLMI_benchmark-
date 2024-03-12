from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from tensorflow.python.keras.applications import ResNet50
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense
    from tensorflow.python.keras import optimizers
    from keras.applications.resnet50 import preprocess_input
    from keras.preprocessing.image import ImageDataGenerator
    from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
    import torch

# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.


class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'Resnet'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {}

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = []

    def set_objective(self, X, y):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.

        # Fixed for our Cancer/not Cancer classes
        self.NUM_CLASSES = 2

        # Fixed for color images
        self.CHANNELS = 1

        self.IMAGE_RESIZE = 224
        self.RESNET50_POOLING_AVERAGE = 'avg'
        self.DENSE_LAYER_ACTIVATION = 'softmax'
        self.OBJECTIVE_FUNCTION = 'categorical_crossentropy'

        # Common accuracy metric for all outputs, but can use different
        # metrics for different output
        self.LOSS_METRICS = ['balanced_accuracy']

        # EARLY_STOP_PATIENCE must be < NUM_EPOCHS
        self.NUM_EPOCHS = 10
        self.EARLY_STOP_PATIENCE = 3
        # Training images processed in each step would be no.-of-train-images
        # / STEPS_PER_EPOCH_TRAINING
        self.STEPS_PER_EPOCH_TRAINING = 10
        self.STEPS_PER_EPOCH_VALIDATION = 10

        # These steps value should be proper FACTOR of no.-of-images in train
        # & valid folders respectively
        # NOTE that these BATCH* are for Keras ImageDataGenerator batching to
        # fill epoch step input
        self.BATCH_SIZE_TRAINING = 100
        self.BATCH_SIZE_VALIDATION = 100

        # Using 1 to easily manage mapping between test_generator & prediction
        # for submission preparation
        self.BATCH_SIZE_TESTING = 1

        self.resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

        # Still not talking about our train/test data or any pre-processing.

        self.model = Sequential()

        # 1st layer as the lumpsum weights from
        # resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
        # NOTE that this layer will be set below as NOT TRAINABLE, i.e.,
        # use it as is
        self.model.add(ResNet50(include_top=False,
                                pooling=self.RESNET50_POOLING_AVERAGE,
                                weights=self.resnet_weights_path))

        # 2nd layer as Dense for 2-class classification, i.e., dog or cat
        # using SoftMax activation
        self.model.add(Dense(self.NUM_CLASSES,
                             activation=self.DENSE_LAYER_ACTIVATION))

        # Say not to train first layer (ResNet) model as it is already trained
        self.model.layers[0].trainable = False

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd,
                           loss=self.OBJECTIVE_FUNCTION,
                           metrics=self.LOSS_METRICS)

        image_size = self.IMAGE_RESIZE

        # preprocessing_function is applied on each image but only after re-sizing & augmentation (resize => augment => pre-process)
        # Each of the keras.application.resnet* preprocess_input MOSTLY mean BATCH NORMALIZATION (applied on each batch) stabilize the inputs to nonlinear activation functions
        # Batch Normalization helps in faster convergence
        data_generator = ImageDataGenerator(preprocessing_function=self.preprocess_input)


    # These steps value should be proper FACTOR of no.-of-images in train
    # & valid folders respectively
        X_tensor = torch.tensor(X)
        self.X, self.y = X_tensor, y

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        self.clf.fit(self.X, self.y)

    def get_next(self, n_iter):
        return n_iter + 1

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(model=self.clf)
