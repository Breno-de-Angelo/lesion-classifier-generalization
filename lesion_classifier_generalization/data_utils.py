"""
Módulo com utilitários para carregamento e preparação de dados
"""

import os
import torch
from torch.utils.data import DataLoader
from lesion_classifier_generalization.pad_ufes_dataset import PADUFES20Dataset
from lesion_classifier_generalization.isic_dataset import ISICDataset


def load_pad_ufes_dataset(data_dir, metadata_file, img_size, test_size=0.2, val_size=0.2, desired_classes=None):
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
    dataset = PADUFES20Dataset(data_dir, metadata_file, img_size, desired_classes)
    
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
        dataset: Dataset ISIC ou PAD-UFES-20
        split_data: Dados divididos
        batch_size: Tamanho do batch
        num_workers: Número de workers para carregamento
    
    Returns:
        tuple: (train_loader, val_loader, test_loader) - test_loader pode ser None
    """
    # Criar datasets
    train_dataset, val_dataset, test_dataset = dataset.get_datasets(split_data)
    
    print(f"Train: {len(train_dataset)} imagens")
    print(f"Val: {len(val_dataset)} imagens")
    
    if test_dataset is not None:
        print(f"Test: {len(test_dataset)} imagens")
    else:
        print("Test: Dataset de teste oficial não disponível")
    
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
    
    # Criar test_loader apenas se o dataset de teste estiver disponível
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        test_loader = None
    
    return train_loader, val_loader, test_loader


def get_dataset_info(dataset, split_data):
    """
    Obtém informações sobre o dataset
    
    Args:
        dataset: Dataset PAD-UFES-20 ou ISIC
        split_data: Dados divididos
    
    Returns:
        dict: Informações do dataset
    """
    train_dataset, val_dataset, test_dataset = dataset.get_datasets(split_data)
    
    dataset_info = {
        'classes': dataset.label_encoder.classes_.tolist(),
        'num_classes': len(dataset.label_encoder.classes_),
        'dataset_sizes': {
            'train': len(train_dataset),
            'val': len(val_dataset)
        }
    }
    
    # Adicionar informações de teste se disponível
    if test_dataset is not None:
        dataset_info['dataset_sizes']['test'] = len(test_dataset)
    else:
        dataset_info['dataset_sizes']['test'] = 0
    
    return dataset_info


def load_isic_dataset(data_dir_2019, data_dir_2020, metadata_2019, metadata_2020, 
                     img_size, val_size=0.2, desired_classes=None, 
                     test_metadata_2019=None, test_metadata_2020=None):
    """
    Carrega e prepara os datasets ISIC 2019 e 2020
    
    Args:
        data_dir_2019: Diretório com as imagens do ISIC 2019
        data_dir_2020: Diretório com as imagens do ISIC 2020
        metadata_2019: Arquivo de metadados de treino do ISIC 2019
        metadata_2020: Arquivo de metadados de treino do ISIC 2020
        img_size: Tamanho das imagens
        val_size: Proporção para validação (dos dados de treino)
        desired_classes: Lista de classes desejadas (opcional)
        test_metadata_2019: Arquivo de metadados de teste do ISIC 2019 (opcional)
        test_metadata_2020: Arquivo de metadados de teste do ISIC 2020 (opcional)
    
    Returns:
        tuple: (dataset, split_data, num_classes, classes)
    """
    # Verificar se os dados existem
    if not os.path.exists(data_dir_2019):
        raise FileNotFoundError(f"Diretório {data_dir_2019} não encontrado!")
    
    if not os.path.exists(data_dir_2020):
        raise FileNotFoundError(f"Diretório {data_dir_2020} não encontrado!")
    
    if not os.path.exists(metadata_2019):
        raise FileNotFoundError(f"Arquivo {metadata_2019} não encontrado!")
    
    if not os.path.exists(metadata_2020):
        raise FileNotFoundError(f"Arquivo {metadata_2020} não encontrado!")
    
    print("Carregando datasets ISIC 2019 e 2020...")
    
    # Carregar dataset com dados de teste se fornecidos
    dataset = ISICDataset(data_dir_2019, data_dir_2020, metadata_2019, metadata_2020, 
                         img_size, desired_classes, test_metadata_2019, test_metadata_2020)

    # Dividir dados de treino em train/val
    print("Dividindo dados de treino em train/val...")
    split_data = dataset.split_data(val_size=val_size)
    
    # Obter informações
    num_classes = len(dataset.label_encoder.classes_)
    classes = dataset.label_encoder.classes_
    
    print(f"Número de classes: {num_classes}")
    print(f"Classes: {classes}")
    
    return dataset, split_data, num_classes, classes
