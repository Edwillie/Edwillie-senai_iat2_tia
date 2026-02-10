import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from matplotlib.image import imread

def quantization_demo():
    # 1. Carregar Imagem
    img = imread('chameleon.jpg') 
    
    # Normalizar para [0, 1] se estiver em 0-255
    img = img / 255.0 
    
    # Transformar a matriz da imagem (H x W x 3) em uma lista de pixels (N x 3)
    pixels = img.reshape(-1, 3) 

    # 2. Configurar a SOM (Ex: 4x4 = 16 cores finais)
    # A SOM vai aprender quais são as 16 cores que melhor resumem a foto
    som_size = 32 
    som = MiniSom(x=som_size, y=som_size, input_len=3, sigma=1.0, learning_rate=0.5)
    
    # Inicialização
    som.random_weights_init(pixels)
    print("Iniciando treinamento...")
    
    # 3. Treinar (Muito mais rápido e simples)
    som.train_random(pixels, 5000) # 5000 iterações
    print("Treinamento concluído.")

    # 4. Quantização (O passo mágico)
    # Para cada pixel original, descobre qual neurônio (cor) é o vencedor
    qnt = som.quantization(pixels) 
    
    # 5. Reconstruir a imagem
    # Reconstrói a imagem usando apenas as cores aprendidas
    clustered = qnt.reshape(img.shape)

    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Quantizada ({som_size*som_size} cores)")
    plt.imshow(clustered)
    plt.axis('off')
    
    plt.show()


quantization_demo()