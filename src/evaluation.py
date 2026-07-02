import pandas as pd
from sklearn.metrics import accuracy_score
from typing import Callable

def evaluate_baseline_on_folds(
    df: pd.DataFrame,
    folds: list[dict],
    baseline_function: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
) -> pd.DataFrame:
    """
    Evaluate a prediction baseline across multiple temporal folds.

    For each fold, the function:

    1. creates chronological training and validation subsets;
    2. checks that both subsets contain observations;
    3. applies the selected baseline function;
    4. validates the structure and values of the returned predictions;
    5. computes the classification accuracy and descriptive fold
       statistics;
    6. stores the results in a summary DataFrame.

    Args:
        df:
            Complete labeled training dataset.

            The DataFrame must contain:

            - ``TS``: temporal identifier;
            - ``class``: binary target;
            - ``ROW_ID``: unique observation identifier.

        folds:
            List of temporal fold dictionaries.

            Each dictionary must contain:

            - ``train_start``;
            - ``train_end``;
            - ``valid_start``;
            - ``valid_end``.

        baseline_function:
            Callable baseline receiving a training DataFrame and a
            validation DataFrame.

            The callable must return a DataFrame containing all original
            validation observations, in the same order, with an
            additional binary column named ``prediction``.

    Returns:
        A DataFrame containing one row per temporal fold with:

        - fold number;
        - training period boundaries;
        - validation period boundaries;
        - positive-class rates;
        - validation accuracy;
        - number of validation predictions;
        - number of correct predictions.

    Raises:
        ValueError:
            If no temporal fold is provided.

        KeyError:
            If a fold dictionary does not contain all required temporal
            boundary keys.

        ValueError:
            If ``df`` does not contain ``TS``, ``class`` and ``ROW_ID``.

        ValueError:
            If a training or validation subset is empty.

        ValueError:
            If the baseline does not return a pandas DataFrame.

        ValueError:
            If the returned DataFrame does not contain a ``prediction``
            column.

        ValueError:
            If the baseline changes the number, identifiers or order of
            validation observations.

        ValueError:
            If predictions contain missing values or values outside the
            binary set ``{0, 1}``.

    Notes:
        The function is dedicated to deterministic baseline strategies.

        Learned machine-learning models will later require a separate
        evaluation function because they introduce an explicit fitting
        phase, preprocessing steps and potentially probability
        predictions.
    """

    required_keys = ["train_start","train_end","valid_start","valid_end"]
    classes = {0, 1}

    if not folds:
        raise ValueError("Array of folds is empty!")
    for fold in folds:
        if not all(key in fold for key in required_keys):
            raise KeyError("All keys are required !")

    results = []
    required_cols = ["TS", "class", "ROW_ID"]

    if not all(col in df.columns for col in required_cols):
        raise ValueError("Missing required columns: TS, class, ROW_ID.") 
    for i in range(len(folds)):
        fold = folds[i]
        fold_train = df[(df["TS"] >= fold["train_start"]) & (df["TS"] <= fold["train_end"])]
        fold_valid = df[(df["TS"] >= fold["valid_start"]) & (df["TS"] <= fold["valid_end"])]
        if fold_train.empty or fold_valid.empty:
            raise ValueError("Training or validation DataFrame is empty.")
        else:
            fold_train_positive_rate = (fold_train["class"] == 1).mean()
            fold_valid_positive_rate = (fold_valid["class"] == 1).mean()
            fold_valid = fold_valid.copy()
            validation_length_before = len(fold_valid)
            validation_id_before = fold_valid["ROW_ID"].copy()
            fold_valid = baseline_function(fold_train,fold_valid)
        if not isinstance(fold_valid, pd.DataFrame):
            raise ValueError("This baseline does not return a DataFrame.")
        if "prediction" not in fold_valid.columns:
            raise ValueError("baseline_function must return a DataFrame with a 'prediction' column.")
        if  (not validation_id_before.equals(fold_valid["ROW_ID"])) or (validation_length_before != len(fold_valid)):
            raise ValueError("The baseline changed the number or order of validation observations.")
        if fold_valid["prediction"].isnull().sum() != 0:
            raise ValueError("You have null values in predictions.")
        if not all(pred in classes  for pred in fold_valid["prediction"].unique()):
            raise ValueError("Prediction is not 0 or 1.")
        
        fold_accuracy = accuracy_score(fold_valid["class"],fold_valid["prediction"])
        fold_n_correct_predictions = (fold_valid["class"] == fold_valid["prediction"]).sum()
        fold_n_valid_predictions = len(fold_valid)
        results.append({
            "fold" : i + 1,
            "train_start" : fold["train_start"],
            "train_end" : fold["train_end"],
            "valid_start" : fold["valid_start"],
            "valid_end" : fold["valid_end"],
            "train_positive_rate" : fold_train_positive_rate,
            "valid_positive_rate" : fold_valid_positive_rate,
            "accuracy" : fold_accuracy,
            "n_valid_predictions" : fold_n_valid_predictions,
            "n_correct_predictions" : fold_n_correct_predictions
        })

    return pd.DataFrame(results)