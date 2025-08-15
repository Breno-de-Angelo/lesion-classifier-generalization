"""
Módulo com definições de modelos para classificação de lesões
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNet(nn.Module):
    """
    Modelo EfficientNet com integração de metadados clínicos
    """
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()

        # Carregar EfficientNet pré-treinado
        try:
            # Para versões mais recentes do torchvision
            self.efficientnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        except AttributeError:
            # Fallback para versões mais antigas
            self.efficientnet = models.efficientnet_b3(pretrained=True)

        # Remover o classificador final
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(1536, num_classes)
        )

    def forward(self, x):
        # Extrair features do EfficientNet
        output = self.efficientnet(x)
        return output


def create_efficientnet_model(num_classes, dropout_rate=0.5):
    """
    Cria modelo EfficientNet
    
    Args:
        num_classes: Número de classes para classificação
        dropout_rate: Taxa de dropout
    
    Returns:
        EfficientNet: Modelo EfficientNet configurado
    """
    model = EfficientNet(
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    return model


def get_model_info(model):
    """
    Obtém informações sobre o modelo
    
    Args:
        model: Modelo PyTorch
    
    Returns:
        dict: Informações do modelo
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_name': model.__class__.__name__
    }
