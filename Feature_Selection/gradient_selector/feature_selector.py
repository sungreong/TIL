import logging

_logger = logging.getLogger(__name__)


class FeatureSelector():

    def __init__(self, **kwargs):
        self.selected_features_ = None
        self.X = None
        self.y = None


    def fit(self, X, y, **kwargs):
        """
        Fit the training data to FeatureSelector
        Paramters
        ---------
        X : array-like numpy matrix
            The training input samples, which shape is [n_samples, n_features].
        y: array-like numpy matrix
            The target values (class labels in classification, real numbers in
            regression). Which shape is [n_samples].
        """
        self.X = X
        self.y = y


    def get_selected_features(self):
        """
        Fit the training data to FeatureSelector
        Returns
        -------
        list :
                Return the index of imprtant feature.
        """
        return self.selected_features_