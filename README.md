# Rede neural VGG16 com Batch Normalization

Utiliza o dataset MNIST-JPG, contendo 60 mil imagens de números de 0 a 9.

## Arquitetura do Modelo
O modelo implementado é uma variação baseada na **VGG16-BN** (VGG16 com *Batch Normalization*), construída em PyTorch.

* **Camadas Convolucionais:** 13 camadas distribuídas em 5 blocos de extração de características (`feature extraction`).
* **Normalização:** `BatchNorm2d` aplicado após cada convolução.
* **Classificador (Dense):** 3 camadas lineares (`Linear`) totalizando 4096 neurónios por camada oculta, protegidas por `Dropout (p=0.5)` para prevenir o *overfitting*.

## Otimizações e Aceleração de Hardware
* **Mixed Precision Training (FP16):** Utilização nativa do `torch.amp.autocast` e `GradScaler` para transferir a carga matemática principal para os Tensor Cores, acelerando o cálculo das matrizes e otimizando o uso da memória VRAM.
* **Otimizador:** Adam Optimizer com a *learning rate* em `1e-4`.

## O "cérebro" da rede é salvo automaticamente ao final do treinamento
* torch.save(model.state_dict(), "pesoscnn.pth")
