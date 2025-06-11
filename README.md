Trabajo Práctico N.º 5 - Modelos Probabilísticos: N-Gramas

**Nombre:** Coral Tolazzi  
**Materia:** Procesamiento del Lenguaje Natural  
**Tema:** Modelos y Clasificación  
**Profesora:** Yanina Ximena Scudero  
**Cuatrimestre:** 1.º Cuatrimestre del 2025  
**Instituto:** Instituto Tecnológico Beltrán

Descripción

Este trabajo práctico implementa un modelo de procesamiento de lenguaje natural basado en n-gramas (bi-gramas y tri-gramas) utilizando Python y NLTK. Se analiza un corpus real con opiniones de estudiantes sobre educación superior en Colombia.

Funcionalidades

- Lectura y procesamiento de un corpus de texto.
- Limpieza y lematización del texto.
- Eliminación de stopwords y signos de puntuación.
- Extracción de 2-gramas y 3-gramas frecuentes (min_df = 2).
- Visualización de los n-gramas más comunes con gráficos de barras.

Componentes principales

- `run()`: función principal que coordina el proceso completo.
- `obtener_corpus()`: lectura del archivo corpus.
- `limpiar_corpus()`: tokeniza, limpia y lematiza cada oración.
- `n_grama()`: extrae y cuenta n-gramas frecuentes.
- `graficar_comparacion_ngrams()`: muestra gráfico con los 10 n-gramas más frecuentes.

Requisitos

Asegúrate de tener instaladas las siguientes dependencias antes de ejecutar el código:

```python
nltk.download('punkt')
nltk.download('stopwords')
nltk
