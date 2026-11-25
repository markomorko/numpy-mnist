import hashlib
import urllib.request
from pathlib import Path

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"

type ImagesAndLabels = tuple[np.ndarray, np.ndarray]


def load_mnist(cache_dir: Path = Path("data")) -> tuple[ImagesAndLabels, ImagesAndLabels]:
    """Download MNIST into `cache_dir` once, check it, and return its (train, test) splits."""
    cache_dir.mkdir(exist_ok=True)
    path = cache_dir / "mnist.npz"
    if not path.exists():
        urllib.request.urlretrieve(MNIST_URL, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != MNIST_SHA256:
        path.unlink()
        raise ValueError("The download is corrupted and has been deleted: run the cell again")
    with np.load(path) as mnist:
        return (mnist["x_train"], mnist["y_train"]), (mnist["x_test"], mnist["y_test"])


(X_train, y_train), (X_test, y_test) = load_mnist() 


type(X_train), X_train.dtype, X_train.shape
y_train

def print_pixel_values(image: np.ndarray) -> None:
    for row in image:
        print("".join(f"{pixel:4}" for pixel in row))


print_pixel_values(X_train[0])

##
import matplotlib.pyplot as plt


def plot_image(image: np.ndarray) -> None:
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    plt.show()


plot_image(X_train[0])

##

y_train[0]


def plot_first_100(images: np.ndarray) -> None:
    for index, image in enumerate(images[:100]):
        plt.subplot(10, 10, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


plot_first_100(X_train)

"""
Ejercicio1 1. Create matrix_first_100_labels, a 10×10 array with the first 100 labels, 
and display it. It should match the grid of images above.
"""
print("Ejercicio 1\n")
# Selecciono los primeros 100 elementos del array de imágenes
matrix_first_100_labels=y_train[:100].reshape(10,10)

# Los muestro                     
print (matrix_first_100_labels.reshape(10,10))

"""
2. Many models expect each example as one flat row of numbers:
 each 28×28 image becomes a vector of 784 pixels. 
 Create X_train_flat and X_test_flat, with shapes 60,000×784 and 10,000×784.
"""
print("Ejercicio 2\n")
# Creo los arrays a partir de los existentes y los ajusto con reshape(filas,columnas)

X_train_flat = X_train.reshape(60000,784)
X_test_flat  = X_test.reshape(10000,784)


"""
3. Change plot_image so that it accepts both a 28×28 image
 and a flat row of 784 pixels. Check it with X_train_flat[0].
"""
print("Ejercicio 3\n")
import matplotlib.pyplot as plt

def plot_image(image: np.ndarray) -> None:
    if image.ndim==1:                  
        image=image.reshape(28,28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    plt.show()

plot_image(X_train_flat[0])


"""
4 Check, with one expression and no loop, 
that the bottom-right pixel is 0 in every training image
"""
print("Ejercicio 4\n")

# Con las funciones de indexación de NumPy
# seleccionamos todos los pixel de la esquina inferior
# derecha de las imagenes

es_cero=np.all(X_train[:,-1,-1] == 0)
if es_cero:
	print("Éxito => son cero")
else:
	print ("Fallo => No todos los pixeles son cero")
		

"""
5. Does any digit touch the edge of its frame? 
Store in n_touching_border the number of training images
with ink (a non-zero pixel) in their first or last 
row or column.

"""		
print("Ejercicio 5\n")

# Con un bucle recorremos las imágenes del array
# Comprobamos si son distintas de cero (1) y
# las guardamos.

n_touching_border=0

for imagen in X_train:
    if (np.any(imagen[ 0, :] !=0 ) or
    	np.any(imagen[-1, :] !=0 ) or
        np.any(imagen[ :, 0] !=0 ) or
   		np.any(imagen[ :,-1] !=0)):
	    n_touching_border+=1

print("Hay ",n_touching_border," imágenes que tocan el borde")

"""
En los bordes de las imágenes
El primer índice indica la fila
y el segundo la columna

Bordes:

[ 0, :] Borde Superior  
[-1, :] Borde Inferior
[ :, 0] Borde izquierdo 
[ :,-1] Borde derecho

El truco es recordar que con -1 indicamos la última fila 
o columna de la dimensión

Por ejemplo, esta matriz de coordenadas fila,columna:

 0,0  0,1  0,2
 1,0  1,1  1,2
 2,0  2,1  2,2

 Pasaría a:

  0,0  0,1  0,-1
  1,0  1,1  1, 2
 -1,0  2,1  2,-1

y los dos puntos => : representan todos los
elementos de esa dimensión.

 0, :  => Primera fila y tooodos los primeros elementos de las columnas
-1, :  => Ultima fila y todos los últimos elementos de las columnas
 :, 0  => Todos los elementos de la primera columna
 :,-1  => Todos los elementos de la última columna


Por eso seleccionamos
[ 0, :] Borde Superior  
[-1, :] Borde Inferior
[ :, 0] Borde izquierdo 
[ :,-1] Borde derecho   

"""

"""
6. Compute X_train[0] + X_train[0] and look at its maximum.
Then compute the same sum correctly, and store in n_overflowed_pixels the number
of pixels where the two results differ. Why do they differ?

"""
"""
print("Ejercicio 6\n")

Si sumamos directamente los valores de X_train
Como solo almacena enteros uint8 (máximo hasta 255)
habría o podría haber "desbordamiento", sumas que superasen
ese máximo.
Por eso hay que convertirlo a un tipo de dato que pueda
almacenar ese tamaño
"""
print("Ejercicio 6\n")
print("Suma directa\n")
suma=X_train[0]+X_train[0]
print(suma.max())
print("suma correcta, convertimos a int32 con astype()")
suma_correcta=X_train[0].astype(np.int32)+X_train[0].astype(np.int32)
print(suma_correcta.max())
# Hay otra forma pero esta me gusta más
# Como solo queremos saber cuanto desborda, siempre
# que sea mayor = que 128 desborda
n_overflowed_pixels = np.count_nonzero(X_train[0] >= 128)
# Son matrices no es lo mismo Multiplicar que Sumar!!!!


"""
7. One-hot encoding
Some models expect each label as a one-hot vector: ten numbers,
all 0 except a 1 at the position of the digit.
Label 5 becomes [0, 0, 0, 0, 0, 1, 0, 0, 0, 0].
Create y_train_onehot and y_test_onehot.

"""
print("Ejercicio 7\n")

# Cada dígito se representa como un vector one-hot
# de diez posiciones con un "1" donde corresponde
# 1 se representaria como una matriz:
#1=> [0,1,0,0,0,0,0,0,0,0]
#2=> [0,0,1,0,0,0,0,0,0,0]
#3=> [0,0,0,1,0,0,0,0,0,0]
#x=> [ceros excepto la posición x =>1]


print("Empleamos la función np.eye\\(\\) que crea una matriz identidad de 10x10\n")
y_train_onehot=np.eye(10,dtype=int)[y_train]
y_test_onehot=np.eye(10,dtype=int)[y_test]
print("y_train_onehot=> ",y_train_onehot,end=("\n"))
print("y_test_onehot=> ",y_test_onehot,end=("\n"))

"""
Ejercicio 8. Muestra el mínimo, máximo,
media y desviación estandar de los 
pixels de las imágenes de entrenamiento
"""
print("Ejercicio 8")

# np tiene funciónes para mostrar esos valores
print("Datos de los pixels de las imágenes de entrenamiento\n")
print("Máximo => ",np.max(X_train),
    " Mínino => ",np.min(X_train),
    " Media => ",np.mean(X_train),
    " Desviación estándar => ",np.std(X_train),sep=":::",end=(":::"))

"""
Ejercicio 9. Escala las imágenes de entrenamiento al rango
[0,1] como X_train_norm, tipo float32. Haz el plot de
la primera imagen
"""
print("Ejercicio 9")

# Convertímos los datos con astype y el tipo de dato
X_train_norm=(X_train/255.00).astype(np.float32)

"""
Reutilizo esta función para mostrar el plot

def plot_image(image: np.ndarray) -> None:
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    plt.show()
"""

plot_image(X_train_norm[0])


"""
Ejercicio 10.
Store in pixel_norm_mean and pixel_norm_std the mean and the standard deviation 
of the pixels of X_train_norm. You will meet these two numbers again.

Almacen en => pixel_norm_mean y pixel_norm_std la media y la desviación estándar
de los pixels de X_train_norm. 
Débes hallar otra vez esos dos valores
"""
print("Ejercicio 10")

#Volvemos a calcular mean y std, ahora de X_train_norm (Con los valores normalizados)

pixel_norm_mean= np.mean(X_train_norm)
pixel_norm_std= np.std(X_train_norm)

print("pixel_norm_mean => ",pixel_norm_mean," pixel_norm_std =>",pixel_norm_std)



"""
Ejercicio 11.
Standardise the images: subtract the mean and divide by the standard deviation,
so that the pixels have mean 0 and standard deviation 1.
Create X_train_standardized and X_train_standardized.
Compute μ and σ on the training images only, 
and use those same two numbers for the test images.

Estandariza las imagenes: resta la media y divídela por la desviación estandar,
de forma que los pixels tengan media => 0 y desviación estándar => 1.

Crea X_train_standardized y X_test_standardized.
Calcula mu y sigma solo en las imágenes de entrenamiento y
emplea esos números para las imágenes de test.
"""

print("Ejercicio 11")

# Numpy nos permite vectorizar operaciones en sus elementos, no 
# necesitamos un bucle para estandarizar uno a uno los elementos

# Como con X_train_norm, hacemos X_test_norm
X_test_norm=(X_test/255.00).astype(np.float32)

X_train_standardized=(X_train_norm - pixel_norm_mean) / pixel_norm_std
X_test_standardized=(X_test_norm - pixel_norm_mean) / pixel_norm_std

print("Media estandarizada de X_train ",X_train_standardized.mean(),"\n")
print("Media estandarizada de X_test ",X_test_standardized.mean(),"\n")
print("Comparación\n")
print("Antes  X_train_norm[0,10,10] \n",X_train_norm[0,10,10],"\n")
print("Despues X_train_standardized[0,10,10]\n",X_train_standardized[0,10,10],"\n")

print ("Ejercicio 12")

"""
12. Standardise X_train_norm, the images scaled to [0, 1], the same way, 
as X_train_norm_standardized. Check with np.allclose that it equals X_train_standardized. 
Why must they be equal?
"""

X_train_norm_standardized=(X_train_norm - X_train_norm.mean())/X_train_norm.std()
if np.allclose(X_train_norm_standardized,X_train_standardized):
    print("\nX_train_norm_standardized y X_train_standardized son aproximadamente iguales")
print("\nLos valores son practicamente iguales debido a todos los valores se dividen entre\n"
    "255 por lo tanto ese factor se cancela al estandarizar")

"""
 "Se volvieron a numerar los ejercicios".
Data Augmentation
El aumento de datos consiste en generar nuevos ejemplos de entrenamiento mediante 
transformaciones de los datos originales. Por ejemplo, un dígito desplazado unos
píxeles o al cuál se ha añadido algo de ruido sigue representantdo el mismo número.
Un modelo entrenado con estoss datos aumentados dispone de ejmplos más varidos,
lo que puede mejorar su capacidad de generalización ante datos nuevos.

Las bibliotecas de deep learning permiten realizar estas tansformaciones automáticamente, pero 
en esta práctica se implementarán de forma manual.

Ejercicio 14
Crea una función llamada add_gaussian_noise(images, std) que 
añada a cada píxel un valor de ruido aleatorio procedente de una 
distribuición normal, con media 0 y desviación estándar std.

La función debe devolver imágenes del mismo tipo y rango que 
la entrada (uint8,0-255). 

Crea X_train_noisy con std=50 y haz plot con las primeras 100 imágenes.

¿Qué es el ruido gaussiano (Gaussian Noise)?
Es una alteración aleatoria de los valores de los píxeles siguiendo una distribución normal.

Puede utilizarse para simular condiciones imperfectas de captura de imágenes, 
como las producidas por los sensores de una cámara. 
Esto puede ayudar a entrenar modelos de aprendizaje automático (Machine Learning) más robustos.


"""
def add_gaussian_noise(images: np.ndarray, std: float) -> np.ndarray:
    rng = np.random.default_rng(seed=0)

    noise = rng.normal(0, std, images.shape)

    noisy_images = images.astype(np.float32) + noise
    noisy_images = np.clip(noisy_images, 0, 255)

    return noisy_images.astype(np.uint8)

X_train_noisy = add_gaussian_noise(X_train, std=50)
plot_first_100(X_train_noisy)

"""
Binarización de imágenes
Escribe => binarize_image (image, threshold)

Esta función devuelve 1 cuando un pixel es mayor que threshold y cero en otro caso,
empleando el tipo de datos uint8.
La función debe funcionar con una imagen individual y con un conjunto de imágenes.

Crea => X_train_binarized con un umbral (treshold) = 128 y representa gráficamente
con plot las primeras 100 imágenes.
"""
def binarize_image(image, threshold):
    Xtb=np.where(image>threshold,1,0).astype(np.uint8)
    return Xtb

X_train_binarized=binarize_image(X_train,128)
plot_first_100(X_train_binarized)       

"""
16
Muestra gráficamente las primeras 100 imágenes de X_train_binarized y
X_train_noisy, y compara los resultados. ¿Qué diferencias observas?

"""
X_train_bin_80 = binarize_image(X_train, threshold=80)
plot_first_100(X_train_bin_80)
X_train_bin_180 = binarize_image(X_train, threshold=180)
plot_first_100(X_train_bin_180)

print("Con el umbral más bajo, se descartan menos píxeles, por eso se ven trazos más gruesos")

"""
Ejercicio 17

Crea => X_train_noisy_binarized 
binarizando las imágenes con ruido mediante un umbral de 128 
y representa las primeras 100. Prueba otros umbrales: ¿cuál elimina la mayor cantidad de ruido 
sin deteriorar los dígitos?
"""
print("X_train_noisy_binarized con umbral 128")
X_train_noisy_binarized=binarize_image(X_train_noisy,128)
plot_first_100(X_train_noisy_binarized)

print("X_train_noisy_binarized con umbral 8")
X_train_noisy_binarized=binarize_image(X_train_noisy,8)
plot_first_100(X_train_noisy_binarized)

print("X_train_noisy_binarized con umbral 80")
X_train_noisy_binarized=binarize_image(X_train_noisy,80)
plot_first_100(X_train_noisy_binarized)     

print("X_train_noisy_binarized con umbral 180")
X_train_noisy_binarized=binarize_image(X_train_noisy,180)
plot_first_100(X_train_noisy_binarized)   

print("X_train_noisy_binarized con umbral 240")
X_train_noisy_binarized=binarize_image(X_train_noisy,240)
plot_first_100(X_train_noisy_binarized)   

print("Si el umbral es muy bajo, se observa ruido, si es muy alto se pierden detalles")

"""
Ejercicio 18
Escribe una función shift_image(image, dx, dy) que desplace una imagen dx píxeles hacia la derecha
 (hacia la izquierda si es negativo) y dy píxeles hacia abajo (hacia arriba si es negativo), 
 rellenando con ceros los espacios que queden vacíos.

Representa la primera imagen desplazada 10 píxeles en cada una de las cuatro direcciones, 
utilizando una celda independiente para cada desplazamiento.
"""
def shift_image(image, dx, dy):
    shifted = np.zeros_like(image)

    h, w = image.shape

    x_src_start = max(0, -dx)
    x_src_end = min(w, w - dx)
    y_src_start = max(0, -dy)
    y_src_end = min(h, h - dy)

    if x_src_start < x_src_end and y_src_start < y_src_end:
        shifted[
            y_src_start + dy:y_src_end + dy,
            x_src_start + dx:x_src_end + dx
        ] = image[
            y_src_start:y_src_end,
            x_src_start:x_src_end
        ]

    return shifted
print("No hay desplazamiento")
plot_image(shift_image(X_train[0], 0, 0))   
print("Derecha 10 píxeles")
plot_image(shift_image(X_train[0], 10, 0))
print("Izquierda 10 píxeles")
plot_image(shift_image(X_train[0], -10, 0))
print("Abajo 10 píxeles")
plot_image(shift_image(X_train[0], 0, 10))
print("Arriba 10 píxeles")
plot_image(shift_image(X_train[0], 0, -10))

"""
Ejercicio 19
Crea X_train_shifted_right_10 desplazando todas las imágenes de entrenamiento 10 píxeles hacia la derecha
 y representa las primeras 100. Algunos dígitos aparecen recortados: ¿cuáles y por qué?
"""
X_train_shifted_right_10 = np.array([shift_image(img, 10, 0) for img in X_train])
plot_first_100(X_train_shifted_right_10)

print("Al desplazar los pixeles hacia la derecha, los dígitos con píxeles cercanos se recortan")


"""
Ejercicio 20
Escribe una función find_limits(image) que devuelva (top, bottom, left, right): 
los índices de la primera y última fila y columna que contienen píxeles con valor distinto de cero.

Comprueba que la primera imagen de entrenamiento devuelve (5, 24, 4, 23). Después, represéntala
 dibujando sus límites mediante líneas (plt.axhline y plt.axvline)
"""
def find_limits(image):
    rows, cols = np.where(image > 0)

    top = rows.min()
    bottom = rows.max()
    left = cols.min()
    right = cols.max()

    return top, bottom, left, right

print(find_limits(X_train[0]))
# (5, 24, 4, 23)

top, bottom, left, right = find_limits(X_train[0])

plt.imshow(X_train[0], cmap="gray")
plt.axhline(top, color="red")
plt.axhline(bottom, color="red")
plt.axvline(left, color="blue")
plt.axvline(right, color="blue")
plt.show()

"""
¿Cómo funciona?
- np.where(image > 0) obtiene las coordenadas de todos los píxeles que contienen tinta.
- rows.min() y rows.max() identifican las filas superior e inferior.
- cols.min() y cols.max() identifican las columnas izquierda y derecha.
- plt.axhline() dibuja líneas horizontales y plt.axvline() líneas verticales.
El resultado es un rectángulo delimitador (bounding box) que encierra todos los píxeles del dígito.

Ejercicio 21

Escribe una función crop_image_limits(image) que devuelva la imagen recortada a sus límites. 
¿Qué dimensiones tiene la primera imagen de entrenamiento después del recorte y por qué? 
Recuerda cómo se construyó MNIST

"""
def crop_image_limits(image):
    top, bottom, left, right = find_limits(image)

    return image[top:bottom+1, left:right+1]

cropped = crop_image_limits(X_train[0])

print(cropped.shape)
plot_image(cropped)

print("mnist se construyó con imagenes normalizadas de 20x20 píxeles conservando sus proporciones")
print("Posteriormente se añadieron 4 píxeles de relleno para que "
"todas las imágenes fueran de 28x28 píxeles")

"""
Ejercicio 22. Desplazamiento aleatorio de imágenes

«Crea X_train_shifted, que contenga todas las imágenes de entrenamiento desplazadas 
aleatoriamente en horizontal y vertical, pero sin permitir que ningún píxel del dígito
quede fuera de la imagen. Utiliza find_limits() para determinar cuánto puede desplazarse 
cada dígito

"""
rng = np.random.default_rng(seed=42)

X_train_shifted = np.empty_like(X_train)

for i, image in enumerate(X_train):
    top, bottom, left, right = find_limits(image)

    # Desplazamientos máximos permitidos
    dx = rng.integers(-left, image.shape[1] - right)
    dy = rng.integers(-top, image.shape[0] - bottom)

    X_train_shifted[i] = shift_image(image, int(dx), int(dy))

plot_first_100(X_train_shifted)

"""
### **Ejercicio 22. Desplazamiento aleatorio de imágenes**

**Traducción:**

«Crea `X_train_shifted`, que contenga todas las imágenes de entrenamiento desplazadas 
aleatoriamente en horizontal y vertical, pero sin permitir que ningún píxel del
 dígito quede fuera de la imagen. Utiliza `find_limits()` para determinar cuánto
puede desplazarse cada dígito.»



- `find_limits()` identifica dónde comienza y termina el dígito.
- `rng.integers()` elige desplazamientos aleatorios dentro del espacio disponible.
- `shift_image()` desplaza la imagen sin perder ningún píxel del dígito.
- `np.empty_like()` reserva un array con las mismas dimensiones y tipo de datos que `X_train`.

Por ejemplo, si un dígito ocupa las columnas 4 a 23 de una imagen de 28 píxeles de ancho, 
puede desplazarse **4 píxeles a la izquierda y 4 a la derecha**.

**Concepto:** esto es una técnica de **aumento de datos (Data Augmentation)**. 
Generamos variaciones de las imágenes originales para que un modelo aprenda
 a reconocer los dígitos independientemente de su posición.
"""
rng = np.random.default_rng(seed=42)

X_train_shifted = np.empty_like(X_train)

for i, image in enumerate(X_train):
    top, bottom, left, right = find_limits(image)

    # Desplazamientos máximos permitidos
    dx = rng.integers(-left, image.shape[1] - right)
    dy = rng.integers(-top, image.shape[0] - bottom)

    X_train_shifted[i] = shift_image(image, int(dx), int(dy))

plot_first_100(X_train_shifted)

"""
Ejercicio 23
Ampliación del conjunto de entrenamiento
Crea X_train_augmented concatenando, en este orden, las imágenes originales de entrenamiento,
 las que contienen ruido, las binarizadas, las que contienen ruido y están binarizadas, 
 y las desplazadas aleatoriamente.

Antes de concatenarlas, convierte los dos conjuntos binarizados de valores 0 y 1 a valores 0 y 255,
 para que todas las imágenes compartan el mismo rango.

Crea y_train_augmented con las etiquetas correspondientes. ¿Cuántas imágenes hay?
"""
X_train_augmented = np.concatenate([
    X_train,
    X_train_noisy,
    X_train_binarized * 255,
    X_train_noisy_binarized * 255,
    X_train_shifted
], axis=0)

y_train_augmented = np.tile(y_train, 5)

print(X_train_augmented.shape)
print(y_train_augmented.shape)

print("Esta técnica permite entrenar un modelo con distintas versiones de los"
      "mismos dígitos")

