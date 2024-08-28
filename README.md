# clasificacion_texto_espanol_bert
## Clasificador de texto en español utilizando BERT para categorizar mensajes en solicitudes, quejas y recomendaciones.

Este proyecto implementa un modelo de clasificación de texto utilizando BERT (Bidirectional Encoder Representations from Transformers) para categorizar mensajes en español en tres clases: solicitudes de información, quejas o reclamaciones, y recomendaciones. El objetivo es demostrar la capacidad de utilizar modelos de lenguaje preentrenados para tareas de clasificación de texto en español, incluyendo la preparación de datos, entrenamiento del modelo, evaluación y análisis de resultados.

## Características

- Utiliza un modelo BERT preentrenado en español (dccuchile/bert-base-spanish-wwm-uncased)
- Implementa un sistema de clasificación de tres clases
- Incluye técnicas de preparación y aumento de datos
- Proporciona una evaluación completa del modelo y visualización de resultados
- Explora ejemplos desafiantes para probar los límites del modelo

## Tecnologías utilizadas

- **Python**
- **Jupyter Notebook**
- **Pandas** para manipulación de datos
- **Scikit-learn** para división de datos y métricas de evaluación
- **Simpletransformers** para una fácil implementación de modelos BERT
- **Matplotlib y Seaborn** para visualización de datos

## Requisitos del sistema

- Python 3.7+
- CUDA compatible GPU (recomendado para entrenamiento más rápido)
- Entorno de ejecución con soporte para GPU

## Instalación

1. Clona este repositorio:
   git clone https://github.com/BeaEsparcia/clasificacion_texto_espanol_bert.git
2. Navega al directorio del proyecto:
   cd clasificador-texto-espanol-bert
3. Instala los paquetes requeridos:
   pip install -r requirements.txt
4. Asegúrate de tener Jupyter Notebook instalado: 
   pip install jupyter (como alternativa puedes usar Google Colab)

## Uso

1. Inicia Jupyter Notebook:
   jupyter notebook
2. Abre el archivo Clasificacion_texto_espanol_BERT.ipynb en la interfaz de Jupyter.
3. Ejecuta las celdas del notebook secuencialmente o usa "Run All".
4. Si prefieres usar Google Colab:
   - Ve a Google Colab.
   - Selecciona "Archivo" > "Subir cuaderno" y sube el archivo .ipynb desde tu máquina.
5. Una vez abierto el cuaderno, puedes ejecutar las celdas secuencialmente para ver el proceso y los resultados.

Nota: El conjunto de datos está incluido directamente en el notebook, por lo que no es necesario cargar datos externos. Todas las operaciones, desde la carga de datos hasta la evaluación del modelo, se realizan dentro del notebook.

## Estructura del proyecto

- Clasificacion_texto_espanol_bERT.ipynb: Notebook principal con todo el código del proyecto
- requirements.txt: Lista de paquetes de Python requeridos

## Proceso de desarrollo

## Desafíos iniciales y soluciones

1. **Problema**: Resultados de evaluación pobres (MCC de 0.0, eval_loss de 1.09783935546875).
   **Solución**: Aumento del dataset a 100 ejemplos por categoría.
2. **Problema**: Uso de modelo pre-entrenado en inglés.
   **Solución**: Cambio a un modelo pre-entrenado en español (dccuchile/bert-base-spanish-wwm-uncased).
3. **Advertencia**: Conflicto entre `os.fork()` y JAX.
   **Solución**: Se implementó la siguiente línea de código para resolver el conflicto:
   ```python
   mp.set_start_method('spawn', force=True)

## Mejoras implementadas

- Adición de métricas detalladas: precisión, recall y F1-score para cada clase.
- Revisión de sobreajuste mediante la introducción de ejemplos más desafiantes.

## Evaluación con ejemplos desafiantes

- Precisión con ejemplos desafiantes: 0.47 (47% de clasificaciones correctas).
- Los ejemplos desafiantes incluyen:
     1. Ambigüedad entre categorías.
     2. Complejidad lingüística aumentada.
     3. Contextos mixtos en un solo mensaje.
     4. Sutileza que requiere comprensión profunda del contexto.

## Próximos pasos

1. Ampliar el conjunto de datos con más ejemplos complejos y ambiguos.
2. Ajuste fino de hiperparámetros y técnicas de regularización.
3. Análisis detallado de errores de clasificación.
4. Implementación de técnicas avanzadas de preprocesamiento de texto.
5. Evaluación de modelos más complejos o con arquitecturas diferentes.

## Notas importantes

1. Se recomienda el uso de una GPU compatible con CUDA para un entrenamiento más eficiente.
2. El entorno de ejecución debe tener soporte para GPU para aprovechar al máximo el rendimiento del modelo.

## Contribuciones

Las contribuciones a este proyecto son bienvenidas. Por favor, siéntete libre de abrir un issue o enviar un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo LICENSE.md para más detalles.


   





