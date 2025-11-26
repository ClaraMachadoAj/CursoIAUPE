from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def treinar_modelo():
    """
    Carrega o dataset Iris, divide em treino e teste,
    treina um modelo RandomForest e retorna:
    - modelo treinado
    - X_test (dados de teste)
    - y_test (rótulos reais do teste)
    - nomes das classes (target_names)
    """

    # Carrega o dataset Iris (flores com 4 medidas e 3 classes)
    iris = load_iris()
    X = iris.data      # Features: medidas das flores
    y = iris.target    # Labels: Setosa, Versicolor, Virginica

    # Divide em conjuntos de treino e teste
    # test_size=0.2 → 20% para teste
    # stratify=y garante proporção igual das classes
    # random_state congela a aleatoriedade para garantir repetibilidade
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Cria o modelo supervisionado RandomForest
    modelo = RandomForestClassifier(
        n_estimators=100,  # número de árvores
        random_state=42    # reproducibilidade
    )

    # Treina o modelo usando os dados de treino
    modelo.fit(X_train, y_train)

    # Retorna tudo que os testes precisam
    return modelo, X_test, y_test, iris.target_names