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
matrix_first_100_labels=y_train[:100]

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
print("Ejercicio 6\n")

info="n_overflowed_pixels=np.max(X_train\\[0\\])*2 => Daría error\n Debemos convertir el resultado de np.max que es de tipo uint8 (0 255 bits) a int"
print("Nota ",info,"\n")

n_overflowed_pixels=int(np.max(X_train[0]))*2
print("n_overflowed_pixels => ",n_overflowed_pixels)