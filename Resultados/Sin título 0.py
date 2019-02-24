# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 21:11:57 2019

@author: Francisco Javier
"""

#cv2.destroyAllWindows()
#img= Image.open("colores1.png")
#myhis=img.histogram()
#subplot(333)
#bar(range(256),myhis[0:256], ec = 'r')
#subplot(336)
#bar(range(256), myhis[256:512], ec = 'g')
#subplot(339)
#bar(range(256), myhis[512:768], ec = 'b')
# 
#savefig("myHIsto.png")
#
#re = Image.open("myHIsto.png")
#plt.show()
#import cv2
#import numpy as np
#from matplotlib import pyplot as plt
#from PIL import Image
#from pylab import subplot,bar,savefig
#
#
#img = cv2.imread('colores1.png')
#cv2.imshow('colores1.png', img)
#
#color = ('b','g','r')
#
#for i, c in enumerate(color):
#    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
#    plt.plot(hist, color = c)
#    plt.xlim([0,256])
#
#plt.show()
#Algoritmo de deteccion de colores
#Por Glar3
#www.robologs.net
#
#Detecta objetos verdes y amarillos
#  
#import cv2
#import numpy as np
#from matplotlib import pyplot as plt
#from PIL import Image  
##Iniciar la camara
#captura = cv2.VideoCapture(0)
#  
#while(1):
#      
#    #Capturamos una imagen y la convertimos de RGB -> HSV
#    _, imagen = captura.read()
#    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
#  
#    #Rango de colores detectados:
#    #Verdes:
#    verde_bajos = np.array([49,50,50], dtype=np.uint8)
#    verde_altos = np.array([100, 255, 210], dtype=np.uint8)
#    #Amarillos:
#    amarillo_bajos = np.array([16,76,72], dtype=np.uint8)
#    amarillo_altos = np.array([30, 255, 210], dtype=np.uint8)
#  
#    #Detectar los pixeles de la imagen que esten dentro del rango de verdes
#    mascara_verde = cv2.inRange(hsv, verde_bajos, verde_altos)
#     
#    #Detectar los pixeles de la imagen que esten dentro del rango de amarillos
#    mascara_amarillo = cv2.inRange(hsv, amarillo_bajos, amarillo_altos)
#  
#    #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
#    kernel = np.ones((6,6),np.uint8)
#    mascara_verde = cv2.morphologyEx(mascara_verde, cv2.MORPH_CLOSE, kernel)
#    mascara_verde = cv2.morphologyEx(mascara_verde, cv2.MORPH_OPEN, kernel)
#    mascara_amarillo = cv2.morphologyEx(mascara_amarillo, cv2.MORPH_CLOSE, kernel)
#    mascara_amarillo = cv2.morphologyEx(mascara_amarillo, cv2.MORPH_OPEN, kernel)
# 
#    #Unir las dos mascaras con el comando cv2.add()
#    mask = cv2.add(mascara_amarillo, mascara_verde)
# 
#    #Mostrar la imagen de la webcam y la mascara verde
#    cv2.imshow('verde', mask)
#    cv2.imshow('Camara', imagen)
#    tecla = cv2.waitKey(5) & 0xFF
#    if tecla == 27:
#        break
#
#cv2.destroyAllWindows()

#histograma con ejes x y y
#
#import cv2 
#import numpy as np
#from matplotlib import pyplot as plt
#
#img= cv2.imread("verde.png",0)
#plt.hist(img.ravel(),256,[0 , 256])
#cv2.imshow("Imagen original", img)
#plt.show()
#cv2.waitKey(0)
#cv2.destroyAllWindows()