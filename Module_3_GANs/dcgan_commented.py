# Deep Convolutional GANs

# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Setting some hyperparameters
batchSize = 64 # We set the size of the batch.
imageSize = 64 # We set the size of the generated images (64x64).

# Creating the transformations
# We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.
# Estas transformacines se hacen para el Generator
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) 

#-----------------------------------------------------------------------------

# Loading the dataset
# We download the training set in the ./data folder and we apply the previous transformations on each image.
dataset = dset.CIFAR10(root = './data', download = True, transform = transform) 

# We use dataLoader to get the images of the training set batch by batch.
# shuffle: es para que las baraje.
# num_workers: nuero de hilos paralelos. Así lo cargamos más rápido el dataset.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) 

#------------------------------------------------------------------------------



# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
# Inicializa los pesos de la red neuronal que se le pasa como parámetro.
# Se van a pasar por esta función tanto el generator como el discriminator
def weights_init(m):
    
    #Obtenemos todos los nombres de las clases de la red que se pasa
    classname = m.__class__.__name__
    
    # Si la clase es una Convolucional
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    # Si la clase es Batch Normalization
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Defining the generator
# Esta clase hereda de nn.Module
class G(nn.Module): # We introduce a class to define the generator.

    #Constructor de la clase.
    def __init__(self): # We introduce the __init__() function that will define the architecture of the generator.
        
        # Activamos la herencia e invocamos al constructor de la clase padre.
        super(G, self).__init__() # We inherit from the nn.Module tools.
        
        # We create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.).
        self.main = nn.Sequential( 
            
            # We start with an inversed convolution. -------------------------
            # 100 --> Longitud del vector de entrada.
            # 512 --> Numero de feature maps (filtros convolucionales) de la salida.
            # 4 --> Tamaño de los filtros, será 4x4
            # 1 --> Stride
            # 0 --> padding
            # bias --> No queremos trabajar con bias.
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False), 
            
            # We normalize all the features (512) along the dimension of the batch.
            nn.BatchNorm2d(512), 
            
            # We apply a ReLU rectification to break the linearity.
            nn.ReLU(True), 
            
            #------------------------------------------------------------------
            
            # We add another inversed convolution.-----------------------------
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(256), # We normalize again.
            nn.ReLU(True), # We apply another ReLU.
            
            #------------------------------------------------------------------
                   
            # We add another inversed convolution.
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(128), # We normalize again.
            nn.ReLU(True), # We apply another ReLU.
            
            #------------------------------------------------------------------
            
            # We add another inversed convolution.-----------------------------
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(64), # We normalize again.
            nn.ReLU(True), # We apply another ReLU.
            
            #------------------------------------------------------------------
            
            # We add another inversed convolution.-----------------------------
            # Aqui la salida tiene 3 canales para que sea una imagen
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False), 
            # We apply a Tanh rectification to break the linearity and stay between -1 and +1.
            # Esto lo queremos para que siga el mismo estandard que las imagenes del Dataset.
            nn.Tanh() 
            
            #------------------------------------------------------------------
        )

    # Esta función la hereda de la calse padre y la sobreescribe
    # We define the forward function that takes as argument an input that will 
    # fed to the neural network, and that will return the output containing the generated images.
    def forward(self, input): 
        
        # We forward propagate the signal through the whole neural network of 
        # the generator defined by self.main.
        output = self.main(input) 
        
        return output # We return the output containing the generated images.


# Creating the generator-------------------------------------------------------
netG = G() # We create the generator object.

# La función apply la hereda de nn.Module
netG.apply(weights_init) # We initialize all the weights of its neural network.

#------------------------------------------------------------------------------

# Defining the discriminator

class D(nn.Module): # We introduce a class to define the discriminator.

    def __init__(self): # We introduce the __init__() function that will define the architecture of the discriminator.
        super(D, self).__init__() # We inherit from the nn.Module tools.
        self.main = nn.Sequential( # We create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.).
            nn.Conv2d(3, 64, 4, 2, 1, bias = False), # We start with a convolution.
            nn.LeakyReLU(0.2, inplace = True), # We apply a LeakyReLU.
            nn.Conv2d(64, 128, 4, 2, 1, bias = False), # We add another convolution.
            nn.BatchNorm2d(128), # We normalize all the features along the dimension of the batch.
            nn.LeakyReLU(0.2, inplace = True), # We apply another LeakyReLU.
            nn.Conv2d(128, 256, 4, 2, 1, bias = False), # We add another convolution.
            nn.BatchNorm2d(256), # We normalize again.
            nn.LeakyReLU(0.2, inplace = True), # We apply another LeakyReLU.
            nn.Conv2d(256, 512, 4, 2, 1, bias = False), # We add another convolution.
            nn.BatchNorm2d(512), # We normalize again.
            nn.LeakyReLU(0.2, inplace = True), # We apply another LeakyReLU.
            nn.Conv2d(512, 1, 4, 1, 0, bias = False), # We add another convolution.
            nn.Sigmoid() # We apply a Sigmoid rectification to break the linearity and stay between 0 and 1.
        )

    # We define the forward function that takes as argument an input that will 
    # be fed to the neural network, and that will return the output which will be a value between 0 and 1.
    def forward(self, input): 
        
        # We forward propagate the signal through the whole neural network of 
        # the discriminator defined by self.main.
        output = self.main(input) 
        
        # Para que la salida de una convolución sea un numero, tenemos que hacer un flatten.
        return output.view(-1) # We return the output which will be a value between 0 and 1.

# Creating the discriminator
netD = D() # We create the discriminator object.
netD.apply(weights_init) # We initialize all the weights of its neural network.

# Training the DCGANs ---------------------------------------------------------

# We create a criterion object that will measure the error between the prediction 
# and the target.
criterion = nn.BCELoss() # Binary Cross Entropy --> nos da numeros entre 0 y 1.

# We create the optimizer object of the discriminator.
# Los parametros de la red.
# Learning Rate
# betas --> som parametroe de Adam Optimizer.
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999)) 

# We create the optimizer object of the generator.
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999)) 


# En cada epoch se pasan todas las imagenes del dataset.
for epoch in range(25): # We iterate over 25 epochs.

    # Cogemos todas la imagenes en batches, tak y como nos lo da dataloader.
    for i, data in enumerate(dataloader, 0): # We iterate over the images of the dataset.
        
        # 1st Step: Updating the weights of the neural network of the discriminator
        # We initialize to 0 the gradients of the discriminator with respect to the weights.
        netD.zero_grad() 
        
        # Training the discriminator with a real image of the dataset
        # We get a real image of the dataset which will be used to train the discriminator.
        # El primer elemento son las imagen del batch y el segundo son las etiquetas.
        real, _ = data 
        
        # La pasamos a PyTorch Variales para que sirva de entrada a la red neuronal.
        input = Variable(real) # We wrap it in a variable.
        
        # Generamos los targets que deberian salr de las fits reales, que son todos 1.
        # Para ello creams un tensor de a misma longitud que el batch [0] con todo 1s.
        # para que sea aceptado por PyTorch se mete dentro de una variable.
        target = Variable(torch.ones(input.size()[0])) # We get the target.
        
        # We forward propagate this real image into the neural network of the 
        # discriminator to get the prediction (a value between 0 and 1).
        output = netD(input) 
        
        # We compute the loss between the predictions (output) and the target (equal to 1).
        errD_real = criterion(output, target) 
        
        
        # Training the discriminator with a fake image generated by the generator
        
        # We make a random input vector (noise) of the generator.
        # Creamos un vector de ruido de longitud 100 igual que la entrada del generador
        # El primer argumento es el tamaño del batch.
        # El segundo es la longitud del vector de entrada que es 100 para el generador.
        # El resto de parametros son fake dimensiones para el feature map.
        # Y se pone dentro de una variable para que se acepte por PyTorch.
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1)) 
        
        # We forward propagate this random input vector into the neural network
        # of the generator to get some fake generated images.
        fake = netG(noise) 
        
        # Obtenemos los targets que en este caso ar ser fake images son 0s.
        target = Variable(torch.zeros(input.size()[0])) # We get the target.
        
        # We forward propagate the fake generated images into the neural network 
        # of the discriminator to get the prediction (a value between 0 and 1).
        # Usa el detach para elimnar los gradentes de la variable de salida del
        # generador. Esto hace que ocupe menos memoria y los calculos vayan más rápidos.
        output = netD(fake.detach()) 
        
        # We compute the loss between the prediction (output) and the target (equal to 0).
        errD_fake = criterion(output, target) 

        # Backpropagating the total error
        
        # We compute the total error of the discriminator.
        errD = errD_real + errD_fake 
    
        # We backpropagate the loss error by computing the gradients of the total
        # error with respect to the weights of the discriminator.
        errD.backward() 
        
        # We apply the optimizer to update the weights according to how much they 
        # are responsible for the loss error of the discriminator
        optimizerD.step() .

        # 2nd Step: Updating the weights of the neural network of the generator

        # We initialize to 0 the gradients of the generator with respect to the weights.
        netG.zero_grad() 
        
        # El objetivo es que el discriminador diga que son fotos reales, por lo que
        # son 1s
        target = Variable(torch.ones(input.size()[0])) # We get the target.
        
        # We forward propagate the fake generated images into the neural network
        # of the discriminator to get the prediction (a value between 0 and 1).
        output = netD(fake) 
        
        # We compute the loss between the prediction (output between 0 and 1) 
        # and the target (equal to 1).
        errG = criterion(output, target) 
        
        # We backpropagate the loss error by computing the gradients of the total
        # error with respect to the weights of the generator.
        errG.backward() 
        
        # We apply the optimizer to update the weights according to how much 
        # they are responsible for the loss error of the generator.
        optimizerG.step() 
        
        # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps

        # We print les losses of the discriminator (Loss_D) and the generator (Loss_G).
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data[0], errG.data[0])) 
        
        if i % 100 == 0: # Every 100 steps:
            
            # We save the real images of the minibatch.
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True) 
            
            fake = netG(noise) # We get our fake generated images.
            
            # We also save the fake generated images of the minibatch.
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True) 