def train_test_split_by_cv(X, y_dep, y_sen, meta, test_size=0.2, random_state=42):
    """
    Perform train-test split at CV level to avoid leakage.
    """

    df = pd.concat(
        [X, y_dep, y_sen, meta],
        axis=1
    )

    unique_cvs = df["cv_id"].unique()

    train_cvs, test_cvs = train_test_split(
        unique_cvs,
        test_size=test_size,
        random_state=random_state
    )

    train_df = df[df["cv_id"].isin(train_cvs)]
    test_df = df[df["cv_id"].isin(test_cvs)]

    return (
        train_df["text"], test_df["text"],
        train_df["department"], test_df["department"],
        train_df["seniority"], test_df["seniority"],
    )