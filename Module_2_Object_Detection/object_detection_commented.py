# Object Detection

# Importing the libraries --------------------------------------------------

# Importamos PyTorch: contiene dynamic graphs que van muy bien
# para calcular Gradientes en el backpropagation
import torch
from torch.autograd import Variable

# Aunque no hagamos el modelo con OpenCV lo usaremos para pintar rectangulos
# alrededor del perro y de los humanos.
import cv2

# De la carpeta "data" traemos las siguientes clases
# BaseTransform: Hace las imagenes compatibles con las entradas de las Redes Neuronales 
# VOC_CLASSES: Es un diccionario que codifica las imagenes. p.e: avion se codifica como 1,
# perro se codifica como 2, barco se codifica como 3, ...
from data import BaseTransform, VOC_CLASSES as labelmap

# Importamos la librería donde tenemos el modelo de SSD. Importamos el constructor.
from ssd import build_ssd

# Importamos la librería para procesar las imagenes de video. Es similar a PIL pero
# es mejor en terminos de código.
import imageio

# ---------------------------------------------------------------------------

# Defining a function that will do the detections
# La detección, tal y como se hizo con OpenCV, no se hace sobre el video, sino que 
# hace frame a frame. En este caso no necesitamos una imagen en escala de grises como
# pasaba con OpenCV
# frame: imagen en color.
# net: sera la red neuronal que implemente el SSD
# transform: transformaciones sobre la imagen para que sea compatible con SSD.
def detect(frame, net, transform): # We define a detect function that will take as inputs, a frame, a ssd neural network, and a transformation to be applied on the images, and that will return the frame with the detector rectangle.
    
    #Obtenemos las dimensiones de la imagen. El elemento 1 es la altura, el dos es
    # la anchura y el 3 es el canal de color. Cogemos solo los dos primeros.
    height, width = frame.shape[:2] # We get the height and the width of the frame.
    
    #Ahora tenemos que transformar la imagen para que sea compatible con SSD.
    # La primera tranformación lo pone con las dimensiones correctas y con los colores
    # correctos para el modelo. Solo cogemos el primer elemento que nos devuelve 
    # la función [0].
    frame_t = transform(frame)[0] # We apply the transformation to our frame.
    
    # Pasamos de un numpy array a un tensor de PyTorch. También permutamos los colores
    # de rbg a grb para tenerlos en el orden correcto de PyTorch.
    x = torch.from_numpy(frame_t).permute(2, 0, 1) # We convert the frame into a torch tensor.
    
    # Añadimos una dimensión falsa correspondiente al batch y la convertimos en una variable
    # de PyTorch. Lar redes neuronales no aceptan un solo valor sino que aceptan batches.
    # Hacemos que la primera dimensión sea el batch y las siguientes son la entrada.
    x = Variable(x.unsqueeze(0)) # We add a fake dimension corresponding to the batch.
    
    # Obtenemos el resultado de la red neuronal ya entrenada.
    y = net(x) # We feed the neural network ssd with the image and we get the output y.
    
    # Creamos un tensor de pytorch que contenga la salida.
    # detections = [batch, num. classes, num. ocurrencies of the class,(score, x0, y0, x1, y1)]
    detections = y.data # We create the detections tensor contained in the output y.
    
    # Creamos un tensor que esté con a escala apropiada, valores entre 0 y 1 con 
    # las posiciones de la esquina superior izquierda y la esquina inferior derecha.
    scale = torch.Tensor([width, height, width, height]) # We create a tensor object of dimensions [width, height, width, height].
    
    # Recorremos la lista de clases.
    for i in range(detections.size(1)): # For every class:
        
        j = 0 # We initialize the loop variable j that will correspond to the occurrences of the class.
        
        # We take into account all the occurrences j of the class i that have a matching score larger than 0.6.
        while detections[0, i, j, 0] >= 0.6: 
            
            # We get the coordinates of the points at the upper left and the lower right of the detector rectangle.
            # Y lo pasamos a numpy array para poder pintar el rectangulo con OpenCV
            pt = (detections[0, i, j, 1:] * scale).numpy() 
            
            # Pintamos el rectangulo. Convertimos las coordenadas a enteros.
            # Indicamos el color del rectangulo y el grosor de la lina
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            
            # We put the label of the class right above the rectangle.
            # frame: la imagen original.
            # labelmap:  nos coge del diccionario a partir del codigo el texto de la clase.
            # posición donde queremos poner el texto
            # Tipo de fuente del texto.
            # Tamaño del texto
            # Color del texto
            # Anchura del texto
            # Tipo de linea --> la elegimos continua.
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) 
            
            j += 1 # We increment j to get to the next occurrence.
   
    return frame # We return the original frame with the detector rectangle and the label around the detected object.

# ----------------------------------------------------------------------------


# Creating the SSD neural network
# build_ssd es una funcion que esta en el fichero ssd.py
# El parametro que se le pasa es la fase, que tiene dos posibles valores: train y test.
# Como ya está pre-entrenada lo ponemos en fase de test.
net = build_ssd('test') # We create an object that is our neural network ssd.

# Cargamos los pesos de ssd ya pre-entrenado
# We get the weights of the neural network from another one that is pretrained (ssd300_mAP_77.43_v2.pth).
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) 

#------------------------------------------------------------------------------


# Creating the transformation
# net.size es el tamaño objetivo que queremos tener para las imagenes.
# We create an object of the BaseTransform class, a class that will do the required transformations so that the image can be the input of the neural network.
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) 

#------------------------------------------------------------------------------

# Doing some Object Detection on a video
reader = imageio.get_reader('funny_dog.mp4') # We open the video.

fps = reader.get_meta_data()['fps'] # We get the fps frequence (frames per second).

# Creamos un video de salida con la mismsa frecuencia de frames que la entrada.
# We create an output video with this same fps frequence.
writer = imageio.get_writer('output.mp4', fps = fps) 

# We iterate on the frames of the output video:
for i, frame in enumerate(reader): 
    
    # We call our detect function (defined above) to detect the object on the frame.
    frame = detect(frame, net.eval(), transform) 
    
    writer.append_data(frame) # We add the next frame in the output video.
    
    print(i) # We print the number of the processed frame.
    
writer.close() # We close the process that handles the creation of the output video.