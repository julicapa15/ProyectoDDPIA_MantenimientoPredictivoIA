"""Tests del componente de resultados (app/components/results.py) con AppTest.

`AppTest.from_function` ejecuta el componente en un runtime de Streamlit real
pero sin navegador, así que se puede inspeccionar lo que quedó renderizado.
"""

import json

from streamlit.testing.v1 import AppTest


def _pagina_de_resultados():
    """Script mínimo que renderiza un resultado fijo (corre dentro de AppTest)."""
    import numpy as np

    from app.components.results import render_results
    from app.utils.preprocessing import preprocess_input

    render_results(
        proba=np.array([[0.8, 0.2]]),
        X_new=preprocess_input("L", 298.1, 308.6, 1500.0, 40.0, 0.0),
        alerta="Normal",
        potencia=6283.2,
        delta_t=10.5,
    )


def test_vector_de_entrada_se_muestra_como_json_valido():
    # 1. ARRANGE (Página de resultados con un vector de features de numpy)
    app = AppTest.from_function(_pagina_de_resultados)

    # 2. ACT (Ejecutar el script y leer el bloque de código del vector)
    app.run()
    bloque = app.code[0].value

    # 3. ASSERT (Es JSON parseable, sin el repr `np.float64(...)` de numpy 2)
    assert not app.exception
    assert "np.float64" not in bloque
    assert json.loads(bloque)["Air temperature [K]"] == 298.1
    assert json.loads(bloque)["Type_L"] == 1.0
