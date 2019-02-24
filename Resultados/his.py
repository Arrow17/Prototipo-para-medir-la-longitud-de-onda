# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 02:32:41 2019

@author: Francisco Javier
"""
#%%
import PIL 
from PIL import Image 
from matplotlib import pyplot as plt 
#modulo para generar la grafica
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.misc import *
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import colorsys
from pylab import *
import cv2 
from past import autotranslate

from past.builtins import (str as oldstr, range, reduce, raw_input, xrange)
ruta=("C:/Users/aguil/Desktop/Semestre Enero-Junio/Optoelectronica/Practicas/Práctica 3/histograma1/Resultados" )
rut=("C:/Users/aguil/Desktop/Semestre Enero-Junio/Optoelectronica/Practicas/Práctica 3/histograma1/__pycache__/Cache")
jp= ("C:/Users/aguil/Desktop/Semestre Enero-Junio/Optoelectronica/Practicas/Práctica 3/histograma1/__pycache__/extension")

print("\n\nMateria de optpelectronica. \n\nProcesamiento de imagen. \nFrancisco Javier Mendoza Bautista")
imagen = input("ingrese imagen:  ")

data = imread(imagen)
file=imagen;


c=imagen;
c1= Image.open(c)
c1 = c1.convert("RGB")
c1.save(jp + "/1.jpeg")
img_file = Image.open(jp + "/1.jpeg")
img = img_file.load()

hi = cv2.imread(imagen)


def graficar(datos):
  #plt.figure(1)
    x=range(len(datos))
   # plt.xticks([0, 50, 100, 150, 200, 255],[0, 50, 100, 150, 200, 255])
   # plt.bar(datos, x , align='center', orientation = 'vertical', width=15, linewidth=1)
    plt.figure(3)
    m = max(datos)
    plt.plot(datos,x, 'bo')
    plt.text(222.5, 257.143, m)
    plt.text(280, 257.143, 'es su valor maximo')
    plt.title('Histograma invertido')
    plt.xlabel('Longitud de onda [$\lambda$]')
    plt.ylabel('Cantidad de pixeles')
    plt.savefig(ruta + "/Histograma_invertido.png")
    return None



print (data.shape)
imagen=data[:,:,0]
plt.figure(1)
plt.imshow(data)
plt.colorbar()
plt.title('Imagen normal')
plt.ylabel('Cantidad de pixeles')
plt.xlabel('Cantidad de pixeles')
plt.savefig(ruta + "/Original.png")

foto=Image.open(file)


#si la imagen es a color la convertimos a escala de grises
if foto.mode != 'L':
   foto=foto.convert('L')

foto.save(rut+ "/gray.png")
histograma=foto.histogram()

gris = imread(rut + "/gray.png")
print (gris.shape)

i= gris[:,0:,]
plt.figure(2)
plt.imshow(i,cmap=cm.gray)
plt.colorbar()
plt.title('Imagen en Escala de grises')
plt.ylabel('Cantidad de pixeles')
plt.xlabel('Cantidad de pixeles')
plt.savefig(ruta + "/Escala de grises.png")

graficar(histograma)

plt.figure(4)
plt.plot(histograma)
plt.title('Histograma ')
plt.xlabel('Cantidad de pixeles')
plt.ylabel('Longitud de onda [$\lambda$]')

plt.savefig(ruta + "/Histograma.png")
#------------------------------------------------



# (2) Get image width & height in pixels
[xs, ys] = img_file.size
max_intensity = 100
hues = {}

# (3) Examine each pixel in the image file
for x in xrange(0, xs):
  for y in xrange(0, ys):
    # (4)  Get the RGB color of the pixel
    [r, g, b] = img[x, y]

    # (5)  Normalize pixel color values
    r /= 255.0
    g /= 255.0
    b /= 255.0

    # (6)  Convert RGB color to HSV
    [h, s, v] = colorsys.rgb_to_hsv(r, g, b)

    # (7)  Marginalize s; count how many pixels have matching (h, v)
    if h not in hues:
      hues[h] = {}
    if v not in hues[h]:
      hues[h][v] = 1
    else:
      if hues[h][v] < max_intensity:
        hues[h][v] += 1

# (8)   Decompose the hues object into a set of one dimensional arrays we can use with matplotlib
h_ = []
v_ = []
i = []
colours = []

for h in hues:
  for v in hues[h]:
    h_.append(h)
    v_.append(v)
    i.append(hues[h][v])
    [r, g, b] = colorsys.hsv_to_rgb(h, 1, v)
    colours.append([r, g, b])

# (9)   Plot the graph!
fig = plt.figure(5)
ax = p3.Axes3D(fig)
ax.scatter(h_, v_, i, s=9, c=colours, lw=0)

ax.set_xlabel('Longitud de onda $\lambda$')
ax.set_ylabel('Valor')
ax.set_zlabel('Intensidad')
fig.add_axes(ax)
plt.savefig(ruta + "/Imagen3D.png")



color = ('r','g','b')

for i , col, in enumerate(color):
    plt.figure(6)
    histr = cv2.calcHist([hi],[i],None,[256],[0,256])
    plt.plot(histr,color=col)
    plt.xlim([0,256])

plt.show()


