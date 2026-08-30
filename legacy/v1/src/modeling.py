from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def build_logistic_regression_pipeline(
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: int = 42
    ) -> Pipeline:
    """
    Build a Scikit-Learn pipeline for binary classification using
    Logistic Regression.

    The pipeline performs three consecutive operations:

    1. Missing value imputation using a SimpleImputer.
    2. Feature standardization using a StandardScaler.
    3. Binary classification using LogisticRegression.

    All preprocessing steps are fitted exclusively on the training
    subset of each temporal fold, ensuring that no information from
    the validation period leaks into the training process.

    Parameters
    ----------
    C : float, default=1.0
        Inverse regularization strength. Smaller values correspond to
        stronger regularization.

    solver : str, default="lbfgs"
        Optimization algorithm used to estimate the model parameters.

    max_iter : int, default=1000
        Maximum number of optimization iterations.

    random_state : int, default=42
        Random seed used to ensure reproducibility whenever supported
        by the selected solver.

    Returns
    -------
    Pipeline
        An untrained Scikit-Learn Pipeline composed of:

        - SimpleImputer
        - StandardScaler
        - LogisticRegression

    Notes
    -----
    Missing values are currently replaced by the constant value 0.0.
    This preprocessing choice is considered a baseline strategy and
    may be revisited during future hyperparameter optimization and
    feature engineering experiments.
    """

    classifier = LogisticRegression(        
                C = C,
                solver = solver,
                max_iter = max_iter,
                random_state = random_state
            )
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy="constant",fill_value=0)
    pipeline = Pipeline([
                ('imputer',imputer),
                ('scaler',scaler),
                ('classifier',classifier)
            ])
    
    return pipeline