from utils.evaluation_utils import evaluate_predictions

def run_ml_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_predictions(y_test, y_pred)