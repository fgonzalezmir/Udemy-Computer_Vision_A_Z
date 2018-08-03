# Face Recognition

# Importando las librerias.
import cv2

#-----------------------------------------------------------------------------
# CARGAMOS LOS CASCADES

# Son una serie de filtros que aplicamos uno tras otro para detectar la cara.
# OpenCV tiene un monton de HaarCascade filter para detectar otras cosas como
# sonrisas, ... Están en Github y se pueden descargar.

# We load the cascade for the face.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

# We load the cascade for the eyes.
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 

#-----------------------------------------------------------------------------


#------------------------------------------------------------------------------
# DEFINIMOS LA FUNCIÓN CON LA QUE HACER LAS DETECCIONES

# We create a function that takes as input the image in black and white (gray) 
# and the original image (frame), and that will return the original image with the 
# detector rectangles in face and eyes.
# Cascade funciona solo con imagenes en escala de grises. 
def detect(gray, frame): 
    
    # We apply the detectMultiScale method from the face cascade to locate one
    # or several faces in the image.
    # Parametros de la funcion:
    #    -Imagen en escala de grises.
    #    -Scale Factor: Cuanto aumentamos el tamaño del filtro o cuanto se reduce la imagen.
    #    -Minimo número de vecinos: Numero de zonas vecinas que al menos deben ser aceptadas.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # x, y: Coordenadas de la esquina superior del recuadro.
    # w: Ancho del rectangulo.
    # h: Altura del rectangulo.
    # Iteramos por todas las caras detectadas en la instrucción anterior y dentro de 
    # cada cara detecamos los ojos.
    for (x, y, w, h) in faces: # For each detected face:
        
        # We paint a rectangle around the face.
        # Parametros:
        #     - Imagen
        #     - Coordenadas esquina superios izquierd.
        #     - Coordenadas esquina inferior derecha.
        #     - Color
        #     - Grosor de la linea.
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) 
         
        # Cogemos la región donde está la cara de la imagen de escala de grises.
        roi_gray = gray[y:y+h, x:x+w] # We get the region of interest in the black and white image. 
        
        # Cogemos la región donde está la cara de la imagen original de color.
        roi_color = frame[y:y+h, x:x+w] # We get the region of interest in the colored image.
        
        # We apply the detectMultiScale method to locate one or several eyes in the image.
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3) 
        
        for (ex, ey, ew, eh) in eyes: # For each detected eye
            
            # We paint a rectangle around the eyes, but inside the referential of the face.         
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2) 
    
    return frame # We return the image with the detector rectangles.

#------------------------------------------------------------------------------


# si pones 0 captura desde la webcam del PC.
# si pones 1 lo coge el video de una camara externa
video_capture = cv2.VideoCapture(0) # We turn the webcam on.

while True: # We repeat infinitely (until break):
    
    #Capturamos la ultima imagen de la camara. El primer parametro que nos devuelve lo ignoramos.
    _, frame = video_capture.read() # We get the last frame.
    
    #Obtenemos la imagen en escala de grises.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
    
    canvas = detect(gray, frame) # We get the output of our detect function.
    
    # mostramos el video con los rectangulos
    cv2.imshow('Video', canvas) # We display the outputs.
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break # We stop the loop.


video_capture.release() # We turn the webcam off.
cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.