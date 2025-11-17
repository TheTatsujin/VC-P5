# VC Prática 5
#### Luis Martín Pérez

## Parte 1: Entrenamiento de una red para detectar emociones

Esto se hizo con transfer learning, cogiendo como modelo de base EficientNet. El resultado final no fue muy bueno, así
que se optó por usar simplemente el modelo que `Deepface` incluye por defecto para detección de emociones. Los resultados
tampoco son mucho mejores, sin embargo, parece tener mejor precisión con las pruebas que hice.

### Aplicar filtro que se adapte a las emociones

La solución a este ejercicio se divide en tres partes fundamentales:

1. Detección de caras, con el método `extract_faces` de `Deepface`.
2. Por cada cara detectada, pasar el recorte al modelo clasificador de emociones.
3. Una vez detectada la emoción, aplicar el filtro correspondiente.

Cada filtro consiste en superponer una imagen con un canal alfa sobre la imagen del vídeo.
A continuación se muestra un ejemplo de cada filtro.

| Happy | Sad | Angry | Afraid | Surprised |
|-------|-----|-------|--------|-----------|
|       |     |       |        |           |

## Parte 2: Crear un filtro de cara de temática libre

Para esta parte, se tomó la temática de matrix. La inspiración surgió al dibujar cada punto carácterístico de la cara dado
por `mediapipe`, empezó siendo para depurar y ver lo que podía hacer pero me recordó a los modelos 3D de videojuegos antiguos,
basados en polígonos de baja resolución.

Se muestra un ejemplo:

![](results/matrix_filter.gif)

El código es muy sencillo, se pasa la imagen de la cámara por el model de `medipipe` con `Deepface`  y se 
extraen los puntos característicos (llamados "landmarks"). 

Por otro lado, a la clase del filtro se le pasa la animación del fondo como imagen sobre la que pintar los puntos y se 
consigue este efecto.
