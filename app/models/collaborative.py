"""
Collaborative Filtering — descartado
=====================================
El modelo SVD fue removido porque requiere cientos de interacciones reales
para producir factores latentes significativos. Con el dataset actual introduce
ruido en lugar de mejorar las recomendaciones.

Decisión documentada en benchmarking (IB_Vasquez_Augusto_Reyes_Renato.pdf):
    - El sistema de recomendación usa Content-Based (TF-IDF + coseno).
    - La evaluación de riesgo usa Random Forest (puntaje benchmarking: 3.49).

Si en el futuro se acumulan suficientes interacciones reales (>500),
se puede reimplementar SVD o ALS aquí y reactivarlo en hybrid.py.
"""
