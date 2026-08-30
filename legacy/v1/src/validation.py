def create_expanding_window_folds(dates:list[str],size:int=120,k:int=4) -> list:
    """
    Create expanding-window temporal folds.

    Args:
        dates: Sorted list of unique dates.
        validation_size: Number of dates in each validation fold.
        k: Number of temporal folds to generate.

    Returns:
        A list of dictionaries describing the temporal boundaries of each fold.

    Raises:
        ValueError: If the input arguments are invalid.
    """
        
    if (size <= 0) or (k <= 0) :
        raise ValueError("La taille du set de validation et le nombre de folds doivent etre positifs")
    if (len(dates) == 0) or (len(dates) <= k * size) :
        raise ValueError("Le nombre de dates est insuffisant !")
    if len(set(dates)) != len(dates) :
        raise ValueError("La liste des dates ne doit pas contenir de doublons !")
    dates = sorted(dates)

    dict_enum = {}
    for i in range(0,len(dates)):
        dict_enum[dates[i]] = i

    train_start = dates[0]
    valid_end = dates[-1]
    folds = []
    for i in range(k,0,-1):
        fold_k = {}
        fold_k["fold"] = i
        fold_k["train_start"] = train_start
        indice_valid_start = dict_enum[valid_end] + 1 - size
        valid_start = dates[indice_valid_start]
        train_end = dates[indice_valid_start - 1]
        fold_k["train_end"] = train_end
        fold_k["valid_start"] = valid_start
        fold_k["valid_end"] = valid_end
        valid_end = dates[dict_enum[valid_end] - size]
        folds.append(fold_k)   

    return folds[::-1]

def check_temporal_folds(folds:list[dict],
                         dates:list[str],
                         validation_size:int) -> bool:
    """
    Validate the consistency of temporal folds.

    This function checks that each fold satisfies the temporal validation
    constraints:
        - the training set is not empty;
        - the validation set is not empty;
        - no dates are shared between train and validation;
        - the training period strictly precedes the validation period;
        - each validation fold contains the expected number of dates.

    Args:
        folds: List of temporal folds.
        dates: Sorted list of unique dates.
        validation_size: Expected number of validation dates.

    Returns:
        True if all temporal consistency checks pass.

    Raises:
        ValueError: If at least one temporal constraint is violated.
    """
    
    for i in range(len(folds)):
        train_fold_dates = list(filter(lambda x: (x >= folds[i]["train_start"]) & (x <= folds[i]["train_end"]),dates))
        valid_fold_dates = list(filter(lambda x: (x >= folds[i]["valid_start"]) & (x <= folds[i]["valid_end"]),dates))
        if len(train_fold_dates) == 0:
            raise ValueError(f"L'ensemble d'entrainement du fold {i+1} est vide!")
        if len(valid_fold_dates) == 0:
            raise ValueError(f"L'ensemble de validation du fold {i+1} est vide!")
        if len(set(train_fold_dates) & set(valid_fold_dates)) != 0:
            raise ValueError("Le train et la validation ont des dates en commun !")
        if max(train_fold_dates) >= min(valid_fold_dates):
            raise ValueError(f"{max(train_fold_dates)} n'est pas inférieure à {min(valid_fold_dates)} !")
        if len(valid_fold_dates) != validation_size:
            raise ValueError(f"le nombre de dates du set de validation du fold {i+1} est différent de {validation_size}")
    return True