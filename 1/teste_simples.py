import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def testar():
    df = pd.read_csv("dataset.csv") 
    X = df[["weight", "lenght", "height"]].values 
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.5,  
        stratify=y,
    )
    modelo = Pipeline(steps=[ 
        ("scaler", StandardScaler()), 
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))
    ])
    modelo.fit(X_train, y_train)
    print("Modelo treinado com sucesso.")
    assert modelo.predict([[0.8,1.1,7.5]]) == "small", "Classe prevista é inválida" 