import pytest
from sklearn.metrics import accuracy_score
from modelo import treinar_modelo

def test_acuracia_minima():
    """
    Teste 1: verificar se o modelo treinado alcança
    pelo menos 90% de acurácia no conjunto de teste.
    Isso garante que o modelo generaliza bem.
    """
    modelo, X_test, y_test, target_names = treinar_modelo()

    # Predição feita pelo modelo no conjunto de teste
    y_pred = modelo.predict(X_test)

    # Calcula acurácia
    acuracia = accuracy_score(y_test, y_pred)

    # Verificação: exigimos >= 90%
    assert acuracia >= 0.90, f"Acurácia muito baixa: {acuracia}"


def test_predicao_manual():
    """
    Teste 2: o modelo deve ser capaz de prever
    a classe de uma amostra manual válida.
    Verificamos se o índice retornado está
    dentro do intervalo de classes possíveis.
    """
    modelo, _, _, target_names = treinar_modelo()

    # Amostra de uma flor nova (nunca vista)
    nova_flor = [[5.1, 3.5, 1.4, 0.2]]  # típica de Setosa

    # Predição do modelo
    pred = modelo.predict(nova_flor)[0]

    # Verifica se a classe está no intervalo correto
    assert 0 <= pred < len(target_names), "Classe prevista é inválida"


def test_predicao_tipo_saida():
    """
    Teste 3: checar se a saída do modelo é do tipo numérico.
    Modelos supervisonados de classificação retornam
    inteiros representando as classes.
    """
    modelo, _, _, _ = treinar_modelo()

    # Amostra qualquer válida
    nova_flor = [[6.5, 3.0, 5.2, 2.0]]  # típica de Virginica

    # Predição
    pred = modelo.predict(nova_flor)[0]

    # Verifica se o tipo é inteiro ou float (aceitável)
    assert isinstance(pred, (int, float)), "A saída não é numérica"