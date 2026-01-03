from utils.data_utils import get_baseline_dataset, get_extended_dataset

def data_preparation_pipeline(
    df,
    *,
    extension: bool = False,
    annotated: bool = True
):
    if extension:
        X, y_dep, y_sen, meta = get_extended_dataset(df)
    else:
        X, y_dep, y_sen, meta = get_baseline_dataset(df)

    if not annotated:
        y_dep = None
        y_sen = None

    return X, y_dep, y_sen, meta
