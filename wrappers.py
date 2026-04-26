"""
Recriação das classes auxiliares definidas no Notebook 2, necessárias para
desserializar os arquivos .joblib salvos com FinalModelWithThreshold.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier


def get_proba(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p.ravel()
    s = model.decision_function(X)
    return 1 / (1 + np.exp(-s))


class ProbCalibrator:
    def __init__(self, method, iso=None, lr=None, eps=1e-6):
        self.method = method
        self.iso = iso
        self.lr = lr
        self.eps = float(eps)

    @classmethod
    def fit(cls, method, y, p, eps=1e-6):
        y = np.asarray(y)
        p = np.asarray(p, dtype=float)
        if method == "none":
            return cls(method="none", eps=eps)
        if method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p, y)
            return cls(method="isotonic", iso=iso, eps=eps)
        if method == "sigmoid":
            p_clip = np.clip(p, eps, 1 - eps)
            Xc = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
            lr = LogisticRegression(solver="lbfgs", max_iter=1000)
            lr.fit(Xc, y)
            return cls(method="sigmoid", lr=lr, eps=eps)
        raise ValueError(f"Método de calibração desconhecido: {method}")

    def transform(self, p):
        p = np.asarray(p, dtype=float)
        if self.method == "none":
            return p
        if self.method == "isotonic":
            return self.iso.transform(p)
        if self.method == "sigmoid":
            eps = self.eps
            p = np.clip(p, eps, 1 - eps)
            X = np.log(p / (1 - p)).reshape(-1, 1)
            return self.lr.predict_proba(X)[:, 1]
        raise ValueError(f"Método de calibração desconhecido: {self.method}")

    def __call__(self, p):
        return self.transform(p)


class ProbaCalibratedWrapper:
    def __init__(self, base_estimator, calibrator):
        self.base_estimator = base_estimator
        self.calibrator = calibrator

    def predict_proba(self, X):
        p = get_proba(self.base_estimator, X)
        p_cal = self.calibrator(p)
        return np.column_stack([1.0 - p_cal, p_cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class FinalModelWithThreshold(BaseEstimator, ClassifierMixin):
    def __init__(self, model, threshold, model_name="unknown", metadata=None):
        self.model = model
        self.threshold = float(threshold)
        self.model_name = str(model_name)
        self.metadata = metadata or {}

    def fit(self, X, y, **fit_params):
        self.model.fit(X, y, **fit_params)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return (get_proba(self.model, X) >= self.threshold).astype(int)

    def predict_with_threshold(self, X, threshold=None):
        t = self.threshold if threshold is None else float(threshold)
        p = get_proba(self.model, X)
        return (p >= t).astype(int), p


class SafeCatBoostClassifier(ClassifierMixin, CatBoostClassifier):
    pass


class NominalPreprocessor(BaseEstimator):
    def __init__(self, num_cols, cat_cols, cat_as="string", cat_categories=None,
                 num_strategy="median", cat_strategy="most_frequent",
                 cat_missing_token="MISSING"):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.cat_as = cat_as
        self.cat_categories = cat_categories
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.cat_missing_token = cat_missing_token

    def fit(self, X, y=None):
        X = X.copy()
        self.num_cols_ = list(self.num_cols)
        self.cat_cols_ = list(self.cat_cols)
        self._num_imputer = SimpleImputer(strategy=self.num_strategy).fit(X[self.num_cols_])
        if self.cat_strategy == "constant":
            self._cat_imputer = SimpleImputer(strategy="constant",
                                              fill_value=self.cat_missing_token).fit(
                                                  X[self.cat_cols_].astype("object"))
        else:
            self._cat_imputer = SimpleImputer(strategy=self.cat_strategy).fit(
                X[self.cat_cols_].astype("object"))
        self._cat_categories_ = None
        if self.cat_categories is not None:
            self._cat_categories_ = {
                c: [str(v) for v in self.cat_categories[c]]
                for c in self.cat_cols_ if c in self.cat_categories
            }
        return self

    def transform(self, X):
        # Suporta ambos os nomes
        num_imp = getattr(self, "_num_imputer", None) or getattr(self, "_num_imp", None)
        cat_imp = getattr(self, "_cat_imputer", None) or getattr(self, "_cat_imp", None)
        if num_imp is None or cat_imp is None:
            raise AttributeError(
                "Preprocessor não está fitado: faltam _num_imputer/_num_imp "
                "ou _cat_imputer/_cat_imp."
            )
        Xo = X.copy()
        Xo[self.num_cols_] = num_imp.transform(Xo[self.num_cols_]).astype(float)
        cat = cat_imp.transform(Xo[self.cat_cols_].astype("object"))
        cat_df = pd.DataFrame(cat, columns=self.cat_cols_, index=Xo.index).astype(str)
        if self._cat_categories_:
            for c in self.cat_cols_:
                if c in self._cat_categories_:
                    cat_df[c] = pd.Categorical(cat_df[c],
                                               categories=self._cat_categories_[c])
        if self.cat_as == "category":
            for c in self.cat_cols_:
                cat_df[c] = cat_df[c].astype("category")
        Xo[self.cat_cols_] = cat_df
        return Xo
