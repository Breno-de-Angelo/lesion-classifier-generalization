"""
Módulo de avaliação para modelos de classificação de lesões
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import json
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def evaluate_model(model, test_loader, criterion, device):
    """
    Avalia o modelo no conjunto de teste
    
    Args:
        model: Modelo PyTorch treinado
        test_loader: DataLoader para teste
        criterion: Função de loss
        device: Dispositivo (CPU/GPU)
    
    Returns:
        dict: Dicionário com métricas de avaliação
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    # Calcular métricas
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)

    # Top-2 accuracy
    top2_correct = 0
    for i, (pred, true_label) in enumerate(zip(all_predictions, all_labels)):
        outputs_i = all_outputs[i]
        if outputs_i is not None:
            # Converter para tensor para usar topk
            outputs_tensor = torch.tensor(outputs_i).unsqueeze(0)
            _, top2_indices = outputs_tensor.topk(2)
            if true_label in top2_indices[0]:
                top2_correct += 1

    top2_accuracy = 100. * top2_correct / total if total > 0 else 0

    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions) * 100

    # AUC
    try:
        # One-vs-rest AUC
        auc = roc_auc_score(all_labels, all_predictions, average='macro', multi_class='ovr') * 100
    except:
        auc = 0

    return {
        'loss': avg_loss,
        'accuracy': accuracy / 100,  # Normalizar para 0-1
        'top2_acc': top2_accuracy / 100,
        'balanced_accuracy': balanced_acc / 100,
        'auc': auc / 100
    }


def load_checkpoint(checkpoint_path, model, device):
    """
    Carrega checkpoint do modelo
    
    Args:
        checkpoint_path: Caminho para o checkpoint
        model: Modelo PyTorch
        device: Dispositivo (CPU/GPU)
    
    Returns:
        dict: Informações do checkpoint
    """
    try:
        # Tentar carregar com weights_only=False para compatibilidade
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Modelo carregado com sucesso!")
        return checkpoint
    except Exception as e:
        print(f"Erro ao carregar checkpoint: {e}")
        return None


def save_evaluation_results(test_metrics, dataset_info, save_folder, checkpoint_info=None):
    """
    Salva resultados da avaliação
    
    Args:
        test_metrics: Métricas de teste
        dataset_info: Informações do dataset
        save_folder: Pasta para salvar resultados
        checkpoint_info: Informações do checkpoint (opcional)
    """
    # Converter tipos numpy para tipos Python nativos para serialização JSON
    def convert_numpy_types(obj):
        if hasattr(obj, 'item'):
            # Verificar se é um array de tamanho 1
            if obj.size == 1:
                return obj.item()
            else:
                # Se não for de tamanho 1, converter para lista
                return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(v) for v in obj)
        else:
            return obj
    
    # Converter classes numpy array para lista
    classes_list = dataset_info['classes'].tolist() if hasattr(dataset_info['classes'], 'tolist') else list(dataset_info['classes'])
    
    results = {
        'test_metrics': convert_numpy_types(test_metrics),
        'classes': classes_list,
        'num_classes': int(dataset_info['num_classes']),
        'dataset_sizes': convert_numpy_types(dataset_info['dataset_sizes'])
    }
    
    if checkpoint_info:
        results['checkpoint_info'] = {
            'epoch': checkpoint_info.get('epoch', 'unknown'),
            'val_acc': checkpoint_info.get('val_acc', 'unknown')
        }
    
    # Salvar resultados
    results_path = os.path.join(save_folder, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Resultados salvos em: {results_path}")
    
    return results_path


def log_evaluation_to_wandb(test_metrics, dataset_info):
    """
    Loga resultados de avaliação para o wandb
    
    Args:
        test_metrics: Métricas de teste
        dataset_info: Informações do dataset
    """
    # Log dos resultados básicos para wandb
    wandb.log({
        "test_loss": test_metrics['loss'],
        "test_accuracy": test_metrics['accuracy'],
        "test_top2_accuracy": test_metrics['top2_acc'],
        "test_balanced_accuracy": test_metrics['balanced_accuracy'],
        "test_auc": test_metrics['auc']
    })
    
    # Log das métricas adicionais se disponíveis
    if 'additional_metrics' in test_metrics:
        additional = test_metrics['additional_metrics']
        wandb.log({
            "test_cohen_kappa": additional.get('cohen_kappa', 0),
            "test_macro_precision": additional.get('macro_precision', 0),
            "test_macro_recall": additional.get('macro_recall', 0),
            "test_macro_f1": additional.get('macro_f1', 0),
            "test_weighted_precision": additional.get('weighted_precision', 0),
            "test_weighted_recall": additional.get('weighted_recall', 0),
            "test_weighted_f1": additional.get('weighted_f1', 0)
        })

    # Atualizar summary do wandb
    wandb.run.summary.update({
        "final_test_accuracy": test_metrics['accuracy'],
        "final_test_balanced_accuracy": test_metrics['balanced_accuracy'],
        "final_test_auc": test_metrics['auc'],
        "dataset_size": sum(dataset_info['dataset_sizes'].values()),
        "num_classes": dataset_info['num_classes']
    })
    
    # Atualizar summary com métricas adicionais se disponíveis
    if 'additional_metrics' in test_metrics:
        additional = test_metrics['additional_metrics']
        wandb.run.summary.update({
            "final_test_cohen_kappa": additional.get('cohen_kappa', 0),
            "final_test_macro_f1": additional.get('macro_f1', 0),
            "final_test_weighted_f1": additional.get('weighted_f1', 0)
        })


def print_evaluation_summary(test_metrics):
    """
    Imprime resumo da avaliação
    
    Args:
        test_metrics: Métricas de teste
    """
    print("\n=== RESULTADOS DA AVALIAÇÃO ===")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Top-2 Accuracy: {test_metrics['top2_acc']:.4f}")
    print(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print("=" * 35)
