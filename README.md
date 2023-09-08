# Trabalho Pratico 02
Trabalho Prático 02 - Visão computacional - UFJF

## Como rodar os experimentos

Para criar a imagem Docker navegue até a pasta do projeto e constua a imagem   
`docker build -t tf-gpu .`

Em seguida rode a imagem e chame os scripts   
`docker run -it tf-gpu`

Para rodar cada uma das redes neurais, chame o script passando o dataset (**cifar** ou **fashion**), numero de épocas e tamanho do lote   
`python AlexNet.py cifar 10 128`

Um script em bash está incluso que roda todos os experimentos salvando as imagens em uma pasta chamada figuras  
`./run.sh`

Como houve dificuldade em rodar a rede VGG no meu hardware, um checkpoint da rede (cifar10vgg.h5) também foi incluso e uma novo script que faz uso desse checkpoint
`VGG-checkpoint.py`
