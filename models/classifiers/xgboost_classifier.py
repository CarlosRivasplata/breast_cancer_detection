import xgboost as xgb
from sklearn.model_selection import GridSearchCV

from .base_classifier import BaseClassifier


class XGBoostClassifier(BaseClassifier):
    def __init__(self, search_hyperparameters: bool = False, config=None):
        super().__init__(config)
        self.search_hyperparameters = search_hyperparameters

    def build_model(self):
        return xgb.XGBClassifier(
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        )

    def fit(self, df):
        X = df[self.feature_columns].values
        y = df[self.label_column].map(self.label_map).values

        X_scaled = self.scaler.fit_transform(X)

        if self.search_hyperparameters:
            print("Performing GridSearchCV for XGBoost hyperparameter tuning...")

            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            }

            base_xgb = self.build_model()
            grid = GridSearchCV(
                base_xgb,
                param_grid,
                cv=3,
                scoring='accuracy',
                verbose=1,
                n_jobs=-1
            )
            grid.fit(X_scaled, y)

            print("Best hyperparameters found for XGBoost:")
            print(grid.best_params_)
            print(f"Validation Accuracy (CV): {grid.best_score_:.4f}")

            self.model = grid.best_estimator_
        else:
            self.model = self.build_model()
            self.model.fit(X_scaled, y)
