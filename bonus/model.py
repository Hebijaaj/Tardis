import joblib
import numpy
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score


def main():
    df = pd.read_csv("bonus/cleaned_dataset.csv")

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["month"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["gare_depart", "gare_arrivee"],
            ),
        ]
    )

    x = df[["gare_depart", "gare_arrivee", "month"]]

    y = numpy.array(df["retard_moyen_tous_trains_arrivee"], dtype=numpy.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=43
    )

    r2_results = {}
    pipelines = {}

    pipe_lr = Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])

    pipe_lr.fit(X_train, y_train)

    pred_lr = pipe_lr.predict(X_test)

    r2_score_lr = r2_score(y_test, pred_lr)

    r2_results["Linear Regression"] = r2_score_lr
    pipelines["Linear Regression"] = pipe_lr

    param_grid_lr = {
        "model__fit_intercept": [True, False],
    }

    grid_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=5, scoring="r2", n_jobs=-1)

    grid_lr.fit(X_train, y_train)

    best_lr = grid_lr.best_estimator_
    pred_best_lr = best_lr.predict(X_test)

    r2_best_lr = r2_score(y_test, pred_best_lr)

    r2_results["Tuned Linear Regression"] = r2_best_lr
    pipelines["Tuned Linear Regression"] = best_lr

    pipe_rf = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=43, n_jobs=-1)),
        ]
    )
    # random_state: a random number, n_jobs: use all the cores of the computer

    pipe_rf.fit(X_train, y_train)

    pred_rf = pipe_rf.predict(X_test)

    r2_score_rf = r2_score(y_test, pred_rf)

    r2_results["Random Forest"] = r2_score_rf
    pipelines["Random Forest"] = pipe_rf

    param_grid_rf = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5, 8, None],
        "model__min_samples_split": [2, 5, 10],
    }

    grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring="r2", n_jobs=-1)

    grid_rf.fit(X_train, y_train)

    best_rf = grid_rf.best_estimator_
    pred_best_rf = best_rf.predict(X_test)

    r2_best_rf = r2_score(y_test, pred_best_rf)

    r2_results["Tuned Random Forest"] = r2_best_rf
    pipelines["Tuned Random Forest"] = best_rf

    pipe_gbr = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(learning_rate=0.1, random_state=43)),
        ]
    )
    # learning rate: how much the program ajust its parameters, random_state: also a random number

    pipe_gbr.fit(X_train, y_train)

    pred_gbr = pipe_gbr.predict(X_test)

    r2_score_gbr = r2_score(y_test, pred_gbr)

    r2_results["Gradient Boosting Regressor"] = r2_score_gbr
    pipelines["Gradient Boosting Regressor"] = pipe_gbr

    param_grid_gbr = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5, 8, None],
        "model__min_samples_split": [2, 5, 10],
        "model__learning_rate": [0.05, 0.1, 0.2],
    }

    grid_gbr = GridSearchCV(pipe_gbr, param_grid_gbr, cv=5, scoring="r2", n_jobs=-1)

    grid_gbr.fit(X_train, y_train)

    best_gbr = grid_gbr.best_estimator_
    pred_best_gbr = best_gbr.predict(X_test)

    r2_best_gbr = r2_score(y_test, pred_best_gbr)

    r2_results["Tuned Gradient Boosting Regressor"] = r2_best_gbr
    pipelines["Tuned Gradient Boosting Regressor"] = best_gbr

    choosen_model = max(r2_results, key=r2_results.get)
    print("Choosen model:", choosen_model)

    ## DO THE SAME THING FOR CANCELATION

    x = df[["gare_depart", "gare_arrivee", "month"]]

    df["ratio_annulation"] = (
        (df["nb_annulation"] / df["nb_train_prevu"])
        .replace([float("inf"), -float("inf")], 0)
        .fillna(0)
    )

    y = numpy.array(df["ratio_annulation"], dtype=numpy.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=43
    )

    r2_results = {}
    cancelltion_pipelines = {}

    pipe_lr = Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])

    pipe_lr.fit(X_train, y_train)

    pred_lr = pipe_lr.predict(X_test)

    r2_score_lr = r2_score(y_test, pred_lr)

    r2_results["Linear Regression"] = r2_score_lr
    cancelltion_pipelines["Linear Regression"] = pipe_lr

    param_grid_lr = {
        "model__fit_intercept": [True, False],
    }

    grid_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=5, scoring="r2", n_jobs=-1)

    grid_lr.fit(X_train, y_train)

    best_lr = grid_lr.best_estimator_
    pred_best_lr = best_lr.predict(X_test)

    r2_best_lr = r2_score(y_test, pred_best_lr)

    r2_results["Tuned Linear Regression"] = r2_best_lr
    cancelltion_pipelines["Tuned Linear Regression"] = best_lr

    pipe_rf = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=43, n_jobs=-1)),
        ]
    )
    # random_state: a random number, n_jobs: use all the cores of the computer

    pipe_rf.fit(X_train, y_train)

    pred_rf = pipe_rf.predict(X_test)

    r2_score_rf = r2_score(y_test, pred_rf)

    r2_results["Random Forest"] = r2_score_rf
    cancelltion_pipelines["Random Forest"] = pipe_rf

    param_grid_rf = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5, 8, None],
        "model__min_samples_split": [2, 5, 10],
    }

    grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring="r2", n_jobs=-1)

    grid_rf.fit(X_train, y_train)

    best_rf = grid_rf.best_estimator_
    pred_best_rf = best_rf.predict(X_test)

    r2_best_rf = r2_score(y_test, pred_best_rf)

    r2_results["Tuned Random Forest"] = r2_best_rf
    cancelltion_pipelines["Tuned Random Forest"] = best_rf

    pipe_gbr = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(learning_rate=0.1, random_state=43)),
        ]
    )
    # learning rate: how much the program ajust its parameters, random_state: also a random number

    pipe_gbr.fit(X_train, y_train)

    pred_gbr = pipe_gbr.predict(X_test)

    r2_score_gbr = r2_score(y_test, pred_gbr)

    r2_results["Gradient Boosting Regressor"] = r2_score_gbr
    cancelltion_pipelines["Gradient Boosting Regressor"] = pipe_gbr

    param_grid_gbr = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5, 8, None],
        "model__min_samples_split": [2, 5, 10],
        "model__learning_rate": [0.05, 0.1, 0.2],
    }

    grid_gbr = GridSearchCV(pipe_gbr, param_grid_gbr, cv=5, scoring="r2", n_jobs=-1)

    grid_gbr.fit(X_train, y_train)

    best_gbr = grid_gbr.best_estimator_
    pred_best_gbr = best_gbr.predict(X_test)

    r2_best_gbr = r2_score(y_test, pred_best_gbr)

    r2_results["Tuned Gradient Boosting Regressor"] = r2_best_gbr
    cancelltion_pipelines["Tuned Gradient Boosting Regressor"] = best_gbr

    cancellation_choosen_model = max(r2_results, key=r2_results.get)
    print("Choosen model for cancellation:", cancellation_choosen_model)

    dictionnary = {
        "cancel": cancelltion_pipelines[cancellation_choosen_model],
        "delay": pipelines[choosen_model],
    }
    joblib.dump(dictionnary, "bonus/model.pkl")
