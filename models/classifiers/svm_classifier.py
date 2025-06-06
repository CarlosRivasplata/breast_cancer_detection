from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from .base_classifier import BaseClassifier


class SVMClassifier(BaseClassifier):
    def __init__(self, search_hyperparameters: bool = False, config=None):
        super().__init__(config)
        self.search_hyperparameters = search_hyperparameters

    def build_model(self):
        return SVC(probability=True)

    def fit(self, df):
        X = df[self.feature_columns].values
        y = df[self.label_column].map(self.label_map).values

        X_scaled = self.scaler.fit_transform(X)

        if self.search_hyperparameters:
            print("Performing GridSearchCV for SVM hyperparameter tuning...")

            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }

            base_svm = SVC(probability=True, random_state=42)
            grid = GridSearchCV(
                base_svm,
                param_grid,
                cv=3,
                scoring='accuracy',
                verbose=1,
                n_jobs=-1
            )
            grid.fit(X_scaled, y)

            print("Best hyperparameters found:")
            print(grid.best_params_)
            print(f"Validation Accuracy (CV): {grid.best_score_:.4f}")

            self.model = grid.best_estimator_
        else:
            self.model = self.build_model()
            self.model.fit(X_scaled, y)