"""
Módulo com utilitários para carregamento e preparação de dados
"""

import os
import torch
from torch.utils.data import DataLoader
from lesion_classifier_generalization.pad_ufes_dataset import PADUFES20Dataset


def load_pad_ufes_dataset(data_dir, metadata_file, img_size, test_size=0.2, val_size=0.2):
    """
    Carrega e prepara o dataset PAD-UFES-20
    
    Args:
        data_dir: Diretório com as imagens
        metadata_file: Arquivo de metadados
        img_size: Tamanho das imagens
        test_size: Proporção para teste
        val_size: Proporção para validação
    
    Returns:
        tuple: (dataset, split_data, num_classes, classes)
    """
    # Verificar se os dados existem
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Diretório {data_dir} não encontrado!")
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Arquivo {metadata_file} não encontrado!")
    
    print("Carregando dataset PAD-UFES-20...")
    
    # Carregar dataset
    dataset = PADUFES20Dataset(data_dir, metadata_file, img_size)
    
    # Dividir dados
    print("Dividindo dados em train/val/test...")
    split_data = dataset.split_data(test_size=test_size, val_size=val_size)
    
    # Obter informações
    num_classes = len(dataset.label_encoder.classes_)
    classes = dataset.label_encoder.classes_
    
    print(f"Número de classes: {num_classes}")
    print(f"Classes: {classes}")
    
    return dataset, split_data, num_classes, classes


def create_dataloaders(dataset, split_data, batch_size, num_workers=4):
    """
    Cria dataloaders para treinamento, validação e teste
    
    Args:
        dataset: Dataset PAD-UFES-20
        split_data: Dados divididos
        batch_size: Tamanho do batch
        num_workers: Número de workers para carregamento
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Criar datasets
    train_dataset, val_dataset, test_dataset = dataset.get_datasets(split_data)
    
    print(f"Train: {len(train_dataset)} imagens")
    print(f"Val: {len(val_dataset)} imagens")
    print(f"Test: {len(test_dataset)} imagens")
    
    # Criar dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_dataset_info(dataset, split_data):
    """
    Obtém informações sobre o dataset
    
    Args:
        dataset: Dataset PAD-UFES-20
        split_data: Dados divididos
    
    Returns:
        dict: Informações do dataset
    """
    train_dataset, val_dataset, test_dataset = dataset.get_datasets(split_data)
    
    return {
        'classes': dataset.label_encoder.classes_.tolist(),
        'num_classes': len(dataset.label_encoder.classes_),
        'dataset_sizes': {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'test': len(test_dataset)
        }
    }
