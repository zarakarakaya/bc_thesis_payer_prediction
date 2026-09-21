from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

        self.input_feature_names = None
        self.feature_columns = None

    def fit(self, X, y, feature_names):
        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"X has {X.shape[1]} columns but "
                f"{len(feature_names)} feature names were provided."
            )

        self.input_feature_names = list(feature_names)

        # No feature selection yet, so final features
        # are simply all input features.
        self.feature_columns = list(feature_names)

        self.scaler.fit(X)

        return self

    def transform(self, X):
        return self.scaler.transform(X)

    def fit_transform(self, X, y, feature_names):
        self.fit(X, y, feature_names)
        return self.transform(X)