import numpy as np
from cs231n_kk.assets import optim

class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training classification
    models. The Solver performs stochastic gradient descent using different
    update rules defined in optim.py.
    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.
    To train a model, you will first construct a Solver instance, passing the
    model, dataset, and various options (learning rate, batch size, etc) to the
    constructor. You will then call the train() method to run the optimization
    procedure and train the model.
    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_acc_history and solver.val_acc_history will be lists of the
    accuracies of the model on the training and validation set at each epoch.
    """
    def __init__(self, model, data, **kwargs):
        """
        Construct a new Solver instance.
        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data containing:
            'X_train': Array, shape (N_train, d_1, ..., d_k) of training images
            'X_val': Array, shape (N_val, d_1, ..., d_k) of validation images
            'y_train': Array, shape (N_train,) of labels for training images
            'y_val': Array, shape (N_val,) of labels for validation images
            Optional arguments:
            - update_rule: A string giving the name of an update rule in optim.py.
             Default is 'sgd'.
             - optim_config: A dictionary containing hyperparameters that will be
               passed to the chosen update rule. Each update rule requires different
               hyperparameters (see optim.py) but all update rules require a
               'learning_rate' parameter so that should always be present.
            - lr_decay: A scalar for learning rate decay; after each epoch the
              learning rate is multiplied by this value.
            - batch_size: Size of minibatches used to compute loss and gradient
              during training.
            - num_epochs: The number of epochs to run for during training.
            - print_every: Integer; training losses will be printed every
              print_every iterations.
            - verbose: Boolean; if set to false then no output will be printed
              during training.
            - num_train_samples: Number of training samples used to check training
              accuracy; default is 1000; set to None to use entire training set.
            - num_val_samples: Number of validation samples to use to check val
              accuracy; default is None, which uses the entire validation set.
            - checkpoint_name: If not None, then save model checkpoints here every
              epoch.
        """
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)
        self.num_train_examples = kwargs.pop('num_train_examples', 1000)
        self.num_val_samples = kwargs.pop('num_val_samples', None)
        self.checkpoint_name = kwargs.pop('checkpoint_name', None)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized argument(s) %s" % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
