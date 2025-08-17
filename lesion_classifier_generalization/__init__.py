"""
Lesion Classifier Generalization Package

Um pacote para treinamento e avaliação de modelos de classificação de lesões
com suporte a diferentes datasets e arquiteturas.
"""

from .effnet import EfficientNet, create_efficientnet_model, get_model_info
from .train import train_epoch, validate_epoch, train_model
from .eval import (
    evaluate_model, load_checkpoint, save_evaluation_results,
    log_evaluation_to_wandb, print_evaluation_summary
)
from .data_utils import (
    load_pad_ufes_dataset, load_isic_dataset, create_dataloaders, get_dataset_info
)
from .pad_ufes_dataset import PADUFES20Dataset
from .isic_dataset import ISICDataset

__version__ = "0.1.0"
__author__ = "Lesion Classifier Team"

__all__ = [
    # EfficientNet Models
    "EfficientNet",
    "create_efficientnet_model", 
    "get_model_info",
    
    # Training
    "train_epoch",
    "validate_epoch",
    "train_model",
    
    # Evaluation
    "evaluate_model",
    "load_checkpoint",
    "save_evaluation_results",
    "log_evaluation_to_wandb",
    "print_evaluation_summary",
    
    # Data utilities
    "load_pad_ufes_dataset",
    "load_isic_dataset",
    "create_dataloaders",
    "get_dataset_info",
    
    # Datasets
    "PADUFES20Dataset",
    "ISICDataset"
]
