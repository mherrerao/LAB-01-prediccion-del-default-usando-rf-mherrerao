# Cargamos las librerías
import os
import pandas as pd
import gzip
import json
import pickle
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score

 # Cargar los archivos
train_data = pd.read_csv("files/input/train_data.csv.zip", index_col=False)
test_data = pd.read_csv("files/input/test_data.csv.zip", index_col=False)


# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
train_data.rename(columns={"default payment next month": "default"}, inplace=True)
test_data.rename(columns={"default payment next month": "default"}, inplace=True)

# - Remueva la columna "ID".
train_data.drop(columns="ID", inplace=True)
test_data.drop(columns="ID", inplace=True)


# - Elimine los registros con informacion no disponible.
train_data = train_data.loc[train_data["MARRIAGE"] != 0]
train_data = train_data.loc[train_data["EDUCATION"] != 0]
test_data = test_data.loc[test_data["MARRIAGE"] != 0]
test_data = test_data.loc[test_data["EDUCATION"] != 0]


# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
train_data["EDUCATION"] = train_data["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
test_data["EDUCATION"] = test_data["EDUCATION"].apply(lambda x: 4 if x > 4 else x)



# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.

x_train=train_data.drop(columns= "default")
y_train=train_data["default"]
x_test=test_data.drop(columns= "default")
y_test=test_data["default"]


# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).


def make_pipeline():
    categorical_features = ["EDUCATION", "MARRIAGE", "SEX"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), categorical_features),
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("rf", RandomForestClassifier(random_state=42))
    ])

    return pipeline

# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.

def make_grid_search(pipeline, X_train, y_train):
    param_grid = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [10, None],
    "rf__min_samples_split": [10],
    "rf__min_samples_leaf": [2, 4],
    "rf__max_features": [25]
    }

    model = GridSearchCV(
    pipeline,
    param_grid,
    cv = 10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    verbose=1
    )
    model.fit(X_train, y_train)

    return model


# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.

def save_estimator(estimator):
    models_path = "files/models"
    os.makedirs(models_path, exist_ok=True)
    model_file = os.path.join(models_path, "model.pkl.gz")

    with gzip.open(model_file, "wb") as file:
        pickle.dump(estimator, file)


# Paso 6 Y 7

def save_metrics(model, x_train, y_train, x_test, y_test):
    os.makedirs("files/output", exist_ok=True)

    # Predicciones
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Calcular métricas de confusión
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)


    # Métricas
    resultados = [
        # Métricas train
        {
        'type': 'metrics',
        'dataset': 'train',
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1_score': f1_score(y_train, y_train_pred, zero_division=0)
        },
         # Métricas test
        {
        'type': 'metrics',
        'dataset': 'test',
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_test_pred, zero_division=0)
        },
        # Matriz confusión train
      {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": int(cm_train[0][0]), "predicted_1": int(cm_train[0][1])},
        "true_1": {"predicted_0": int(cm_train[1][0]), "predicted_1": int(cm_train[1][1])}
    },
     # Matriz confusión train
    {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": int(cm_test[0][0]), "predicted_1": int(cm_test[0][1])},
        "true_1": {"predicted_0": int(cm_test[1][0]), "predicted_1": int(cm_test[1][1])}
    }
    ]

    with open("files/output/metrics.json", "w") as file:
        for item in resultados:
            json.dump(item, file)
            file.write("\n")


    return resultados

def main():
        pipeline = make_pipeline()
        model = make_grid_search(pipeline, x_train, y_train)
        save_estimator(model)
        save_metrics(model, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
