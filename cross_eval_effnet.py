"""
Cross-dataset evaluation for EfficientNet models.

- Evaluate PAD-trained model on ISIC
- Evaluate ISIC-trained model on PAD

Ensures label correspondence between datasets (SEK/BCC/NEV/MEL ↔
seborrheic_keratosis/basal_cell_carcinoma/nevus/melanoma).
"""

import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import (
    confusion_matrix, classification_report, balanced_accuracy_score, 
    roc_auc_score, precision_recall_fscore_support, cohen_kappa_score,
    roc_curve, auc
)

from lesion_classifier_generalization import (
    create_efficientnet_model,
    load_isic_dataset,
    load_pad_ufes_dataset,
    load_checkpoint,
)


PAD_TO_ISIC_NAME = {
    "BCC": "basal_cell_carcinoma",
    "MEL": "melanoma",
    "NEV": "nevus",
    "SEK": "seborrheic_keratosis",
}

ISIC_TO_PAD_NAME = {v: k for k, v in PAD_TO_ISIC_NAME.items()}


class RemappedLabelsDataset(Dataset):
    """
    Wraps a dataset and remaps its integer labels using a provided mapping
    (e.g., target_idx -> source_idx) so they align with the source model's head.
    """

    def __init__(self, base_dataset: Dataset, index_mapping: Dict[int, int]):
        self.base_dataset = base_dataset
        self.index_mapping = index_mapping

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label_idx, meta = self.base_dataset[idx]
        mapped_label = self.index_mapping[int(label_idx)]
        return image, mapped_label, meta


def build_index_mapping(source_labels: List[str], target_labels: List[str], mapping: Dict[str, str]) -> Dict[int, int]:
    """
    Build an index mapping so that labels from the target dataset are
    mapped to the indices of the source model's label space.

    Args:
        source_labels: label names (strings) in the source model's training order
        target_labels: label names (strings) in the target dataset's order
        mapping: dict mapping from target label name to source label name (or vice versa)

    Returns:
        Dict[int, int]: mapping target_idx -> source_idx
    """
    source_name_to_idx = {name: i for i, name in enumerate(list(source_labels))}
    index_mapping: Dict[int, int] = {}
    for target_idx, target_name in enumerate(list(target_labels)):
        # Translate the target dataset label name to the source label name
        translated = mapping[target_name]
        # Then find the index of that source label in the source ordering
        if translated not in source_name_to_idx:
            raise ValueError(f"Label '{translated}' not found in source label set {list(source_labels)}")
        index_mapping[target_idx] = int(source_name_to_idx[translated])
    return index_mapping


def create_confusion_matrix(y_true, y_pred, classes, save_path, title_suffix=""):
    """
    Cria e salva matriz de confusão
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições
        classes: Lista de classes
        save_path: Caminho para salvar a imagem
        title_suffix: Sufixo para o título (não usado mais)
    """
    # Mapear nomes completos para siglas
    class_mapping = {
        'basal_cell_carcinoma': 'BCC',
        'melanoma': 'MEL',
        'nevus': 'NEV',
        'seborrheic_keratosis': 'SEK',
        'BCC': 'BCC',
        'MEL': 'MEL',
        'NEV': 'NEV',
        'SEK': 'SEK'
    }
    
    # Converter nomes das classes para siglas
    classes_abbreviated = [class_mapping.get(cls, cls) for cls in classes]
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    
    # Normalizar a matriz
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Criar heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes_abbreviated, yticklabels=classes_abbreviated)
    plt.xlabel('Predição')
    plt.ylabel('Valor Real')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm


def create_classification_report_plot(y_true, y_pred, classes, save_path, title_suffix=""):
    """
    Cria e salva gráfico do relatório de classificação
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições
        classes: Lista de classes
        save_path: Caminho para salvar a imagem
        title_suffix: Sufixo para o título (não usado mais)
    """
    # Mapear nomes completos para siglas
    class_mapping = {
        'basal_cell_carcinoma': 'BCC',
        'melanoma': 'MEL',
        'nevus': 'NEV',
        'seborrheic_keratosis': 'SEK',
        'BCC': 'BCC',
        'MEL': 'MEL',
        'NEV': 'NEV',
        'SEK': 'SEK'
    }
    
    # Converter nomes das classes para siglas
    classes_abbreviated = [class_mapping.get(cls, cls) for cls in classes]
    
    # Calcular métricas por classe
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), average=None
    )
    
    # Criar gráfico de barras com melhor espaçamento
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    
    # Configurações comuns para todos os subplots
    bar_width = 0.6
    x_pos = np.arange(len(classes))
    
    # Precision
    bars1 = ax1.bar(x_pos, precision, width=bar_width, color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1)
    ax1.set_title('Precision por Classe', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(classes_abbreviated, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Recall
    bars2 = ax2.bar(x_pos, recall, width=bar_width, color='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=1)
    ax2.set_title('Recall por Classe', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(classes_abbreviated, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # F1-Score
    bars3 = ax3.bar(x_pos, f1, width=bar_width, color='salmon', alpha=0.8, edgecolor='darkred', linewidth=1)
    ax3.set_title('F1-Score por Classe', fontsize=14, fontweight='bold', pad=20)
    ax3.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1.05)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(classes_abbreviated, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Adicionar valores nas barras com melhor posicionamento
    def add_value_labels(ax, bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            # Posicionar o texto acima da barra com offset adequado
            ax.annotate(f'{value:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Adicionar labels para cada subplot
    add_value_labels(ax1, bars1, precision)
    add_value_labels(ax2, bars2, recall)
    add_value_labels(ax3, bars3, f1)
    
    # Ajustar layout para evitar sobreposição
    plt.tight_layout()
    
    # Salvar com alta qualidade
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_roc_curves(y_true, y_pred_proba, classes, save_path, title_suffix=""):
    """
    Cria e salva curvas ROC para cada classe
    
    Args:
        y_true: Labels verdadeiros
        y_pred_proba: Probabilidades preditas (n_samples, n_classes)
        classes: Lista de classes
        save_path: Caminho para salvar a imagem
        title_suffix: Sufixo para o título (não usado mais)
    """
    # Mapear nomes completos para siglas
    class_mapping = {
        'basal_cell_carcinoma': 'BCC',
        'melanoma': 'MEL',
        'nevus': 'NEV',
        'seborrheic_keratosis': 'SEK',
        'BCC': 'BCC',
        'MEL': 'MEL',
        'NEV': 'NEV',
        'SEK': 'SEK'
    }
    
    # Converter nomes das classes para siglas
    classes_abbreviated = [class_mapping.get(cls, cls) for cls in classes]
    
    plt.figure(figsize=(12, 8))
    
    # Calcular e plotar curva ROC para cada classe
    roc_aucs = []
    for i, class_name in enumerate(classes):
        # Criar labels binários para classe i vs resto
        y_true_binary = (np.array(y_true) == i).astype(int)
        y_score = y_pred_proba[:, i]
        
        # Calcular curva ROC
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        
        # Plotar curva ROC usando a sigla da classe
        class_abbr = classes_abbreviated[i]
        plt.plot(fpr, tpr, lw=2, 
                label=f'{class_abbr} (AUC = {roc_auc:.3f})')
    
    # Plotar linha diagonal (classificador aleatório)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Classificador Aleatório (AUC = 0.500)')
    
    # Configurações do gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
    plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Adicionar texto com AUC médio
    mean_auc = np.mean(roc_aucs)
    plt.text(0.02, 0.98, f'AUC Médio: {mean_auc:.3f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_aucs


def evaluate_model_complete(model, test_loader, criterion, device, classes):
    """
    Avalia o modelo no conjunto de teste em uma única passada
    
    Args:
        model: Modelo PyTorch treinado
        test_loader: DataLoader para teste
        criterion: Função de loss
        device: Dispositivo (CPU/GPU)
        classes: Lista de classes
    
    Returns:
        dict: Dicionário com todas as métricas e resultados
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_outputs = []
    all_image_paths = []
    incorrect_predictions = []
    
    print("Executando avaliação em uma única passada...")
    
    with torch.no_grad():
        for batch_idx, (images, labels, image_paths) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Armazenar todos os resultados
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
            all_image_paths.extend(image_paths if isinstance(image_paths, list) else [str(p) for p in image_paths])
            
            # Identificar predições incorretas
            incorrect_mask = predicted != labels
            for i in range(len(images)):
                if incorrect_mask[i]:
                    incorrect_predictions.append({
                        'image_path': image_paths[i] if isinstance(image_paths[i], list) else str(image_paths[i]),
                        'true_label': classes[labels[i].item()],
                        'predicted_label': classes[predicted[i].item()],
                        'confidence': torch.softmax(outputs[i], dim=0).max().item(),
                        'true_class_idx': labels[i].item(),
                        'predicted_class_idx': predicted[i].item()
                    })
            
            # Progress bar
            if (batch_idx + 1) % 10 == 0:
                print(f"Processados {batch_idx + 1}/{len(test_loader)} batches...")
    
    # Calcular métricas básicas
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    
    # Top-2 accuracy
    top2_correct = 0
    for i, (pred, true_label) in enumerate(zip(all_predictions, all_labels)):
        outputs_i = all_outputs[i]
        if outputs_i is not None:
            outputs_tensor = torch.tensor(outputs_i).unsqueeze(0)
            _, top2_indices = outputs_tensor.topk(2)
            if true_label in top2_indices[0]:
                top2_correct += 1
    
    top2_accuracy = 100. * top2_correct / total if total > 0 else 0
    
    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions) * 100
    
    # AUC - Calcular usando as probabilidades (outputs) em vez de apenas as predições
    try:
        # Converter outputs para probabilidades usando softmax
        outputs_array = np.array(all_outputs)
        outputs_tensor = torch.tensor(outputs_array)
        probabilities = torch.softmax(outputs_tensor, dim=1).numpy()
        
        # Calcular AUC one-vs-rest para cada classe
        auc_scores = []
        for i in range(len(classes)):
            try:
                # Criar labels binários para classe i vs resto
                binary_labels = (np.array(all_labels) == i).astype(int)
                auc_i = roc_auc_score(binary_labels, probabilities[:, i])
                auc_scores.append(auc_i)
            except:
                auc_scores.append(0.0)
        
        # Calcular AUC macro (média das AUCs por classe)
        auc = np.mean(auc_scores) * 100 if auc_scores else 0
    except Exception as e:
        print(f"Aviso: Erro ao calcular AUC: {e}")
        auc = 0
    
    # Métricas adicionais
    additional_metrics = {
        'cohen_kappa': cohen_kappa_score(all_labels, all_predictions),
        'macro_precision': precision_recall_fscore_support(all_labels, all_predictions, average='macro')[0],
        'macro_recall': precision_recall_fscore_support(all_labels, all_predictions, average='macro')[1],
        'macro_f1': precision_recall_fscore_support(all_labels, all_predictions, average='macro')[2],
        'weighted_precision': precision_recall_fscore_support(all_labels, all_predictions, average='weighted')[0],
        'weighted_recall': precision_recall_fscore_support(all_labels, all_predictions, average='weighted')[1],
        'weighted_f1': precision_recall_fscore_support(all_labels, all_predictions, average='weighted')[2]
    }
    
    # Análise de erros
    error_summary = {
        'total_incorrect': len(incorrect_predictions),
        'total_samples': total,
        'error_rate': len(incorrect_predictions) / total,
        'class_errors': {},
        'confusion_pairs': {}
    }
    
    # Análise por classe
    for pred in incorrect_predictions:
        true_class = pred['true_label']
        pred_class = pred['predicted_label']
        
        if true_class not in error_summary['class_errors']:
            error_summary['class_errors'][true_class] = 0
        error_summary['class_errors'][true_class] += 1
        
        # Pares de confusão
        pair = f"{true_class} -> {pred_class}"
        if pair not in error_summary['confusion_pairs']:
            error_summary['confusion_pairs'][pair] = 0
        error_summary['confusion_pairs'][pair] += 1
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy / 100,  # Normalizar para 0-1
        'top2_acc': top2_accuracy / 100,
        'balanced_accuracy': balanced_acc / 100,
        'auc': auc / 100,
        'additional_metrics': additional_metrics,
        'incorrect_predictions': incorrect_predictions,
        'error_summary': error_summary,
        'all_labels': all_labels,
        'all_predictions': all_predictions,
        'all_outputs': all_outputs,
        'all_image_paths': all_image_paths
    }


def save_incorrect_predictions_csv(incorrect_predictions, save_folder):
    """
    Salva lista de predições incorretas em CSV
    
    Args:
        incorrect_predictions: Lista de predições incorretas
        save_folder: Pasta para salvar
    """
    incorrect_file = os.path.join(save_folder, 'incorrect_predictions.csv')
    with open(incorrect_file, 'w') as f:
        f.write("image_path,true_label,predicted_label,confidence,true_class_idx,predicted_class_idx\n")
        for pred in incorrect_predictions:
            f.write(f"{pred['image_path']},{pred['true_label']},{pred['predicted_label']},{pred['confidence']:.4f},{pred['true_class_idx']},{pred['predicted_class_idx']}\n")
    
    print(f"Predições incorretas salvas em: {incorrect_file}")


def save_error_analysis_json(error_summary, save_folder):
    """
    Salva análise de erros em JSON
    
    Args:
        error_summary: Resumo dos erros
        save_folder: Pasta para salvar
    """
    summary_file = os.path.join(save_folder, 'error_analysis.json')
    with open(summary_file, 'w') as f:
        json.dump(error_summary, f, indent=2)
    
    print(f"Análise de erros salva em: {summary_file}")


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


def eval_pad_model_on_isic():
    """Evaluate PAD-trained model on ISIC dataset"""
    print("\n" + "="*60)
    print("EVALUATING PAD-TRAINED MODEL ON ISIC DATASET")
    print("="*60)
    
    # Configs
    DATA_DIR_PAD = "data/pad_ufes_20"
    METADATA_PAD = "data/pad_ufes_20/metadata.csv"
    PAD_CHECKPOINT = "results_pad_ufes_20_wandb/best_model.pth"
    SAVE_FOLDER = "cross_eval_results_pad_on_isic"

    DATA_DIR_2019 = "data/isic2019"
    DATA_DIR_2020 = "data/isic2020"
    METADATA_2019 = "data/isic2019/ISIC_2019_Training_GroundTruth.csv"
    METADATA_2020 = "data/isic2020/ISIC_2020_Training_GroundTruth_v2.csv"

    IMG_SIZE = 224
    BATCH_SIZE = 256
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create save folder
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # Build a PAD dataset object to recover the label ordering used for training
    pad_dataset, pad_split, pad_num_classes, pad_classes = load_pad_ufes_dataset(
        DATA_DIR_PAD, METADATA_PAD, IMG_SIZE, desired_classes=["SEK", "BCC", "NEV", "MEL"]
    )

    # Load ISIC dataset (target) with the four compatible classes
    desired_isic_classes = [
        "melanoma",
        "nevus",
        "basal_cell_carcinoma",
        "seborrheic_keratosis",
    ]
    isic_dataset, isic_split, isic_num_classes, isic_classes, _ = load_isic_dataset(
        DATA_DIR_2019,
        DATA_DIR_2020,
        METADATA_2019,
        METADATA_2020,
        IMG_SIZE,
        desired_classes=desired_isic_classes,
    )

    # Build mapping: ISIC idx -> PAD idx
    isic_to_pad_idx = build_index_mapping(
        source_labels=list(pad_classes),
        target_labels=list(isic_classes),
        mapping=ISIC_TO_PAD_NAME,
    )

    # Build test dataset and wrap with remapping
    _, _, test_dataset = isic_dataset.get_datasets(isic_split)
    remapped_test_dataset = RemappedLabelsDataset(test_dataset, isic_to_pad_idx)
    test_loader = DataLoader(remapped_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Create model with PAD num_classes and load PAD checkpoint
    model = create_efficientnet_model(pad_num_classes).to(DEVICE)
    if not os.path.exists(PAD_CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {PAD_CHECKPOINT}")
    
    checkpoint = load_checkpoint(PAD_CHECKPOINT, model, DEVICE)
    if checkpoint is None:
        raise RuntimeError("Failed to load checkpoint")

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate_model_complete(model, test_loader, criterion, DEVICE, list(pad_classes))

    # Save incorrect predictions
    save_incorrect_predictions_csv(test_metrics['incorrect_predictions'], SAVE_FOLDER)
    
    # Save error analysis
    save_error_analysis_json(test_metrics['error_summary'], SAVE_FOLDER)
    
    # Generate confusion matrix
    y_true = test_metrics['all_labels']
    y_pred = test_metrics['all_predictions']
    
    cm_path = os.path.join(SAVE_FOLDER, 'confusion_matrix.png')
    cm = create_confusion_matrix(y_true, y_pred, list(pad_classes), cm_path)
    
    # Generate classification metrics plot
    metrics_path = os.path.join(SAVE_FOLDER, 'classification_metrics.png')
    create_classification_report_plot(y_true, y_pred, list(pad_classes), metrics_path)
    
    # Generate ROC curves
    y_true_np = np.array(test_metrics['all_labels'])
    outputs_array = np.array(test_metrics['all_outputs'])
    y_pred_proba_np = torch.softmax(torch.tensor(outputs_array), dim=1).numpy()
    roc_path = os.path.join(SAVE_FOLDER, 'roc_curves.png')
    roc_aucs = create_roc_curves(y_true_np, y_pred_proba_np, list(pad_classes), roc_path)
    
    # Create dataset info for saving
    dataset_info = {
        'classes': list(pad_classes),
        'num_classes': pad_num_classes,
        'dataset_sizes': {
            'test': len(test_dataset)
        }
    }

    # Save evaluation results
    save_evaluation_results(test_metrics, dataset_info, SAVE_FOLDER, checkpoint)

    # Print summary
    print_evaluation_summary(test_metrics)
    
    # Print additional metrics
    additional_metrics = test_metrics['additional_metrics']
    print("\n=== MÉTRICAS ADICIONAIS ===")
    print(f"Cohen's Kappa: {additional_metrics['cohen_kappa']:.4f}")
    print(f"Macro Precision: {additional_metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {additional_metrics['macro_recall']:.4f}")
    print(f"Macro F1: {additional_metrics['macro_f1']:.4f}")
    print(f"Weighted Precision: {additional_metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall: {additional_metrics['weighted_recall']:.4f}")
    print(f"Weighted F1: {additional_metrics['weighted_f1']:.4f}")
    print("=" * 35)

    print(f"\nResultados salvos em: {SAVE_FOLDER}")
    print("\n📊 Arquivos gerados:")
    print(f"  - Matriz de confusão: {cm_path}")
    print(f"  - Métricas por classe: {metrics_path}")
    print(f"  - Predições incorretas: {os.path.join(SAVE_FOLDER, 'incorrect_predictions.csv')}")
    print(f"  - Análise de erros: {os.path.join(SAVE_FOLDER, 'error_analysis.json')}")
    print(f"  - Curvas ROC: {roc_path}")
    print(f"  - Resultados completos: {os.path.join(SAVE_FOLDER, 'evaluation_results.json')}")


def eval_isic_model_on_pad():
    """Evaluate ISIC-trained model on PAD dataset"""
    print("\n" + "="*60)
    print("EVALUATING ISIC-TRAINED MODEL ON PAD DATASET")
    print("="*60)
    
    # Configs
    DATA_DIR_PAD = "data/pad_ufes_20"
    METADATA_PAD = "data/pad_ufes_20/metadata.csv"
    PAD_DESIRED = ["SEK", "BCC", "NEV", "MEL"]
    ISIC_CHECKPOINT = "results_isic_2019_2020_wandb/best_model.pth"
    SAVE_FOLDER = "cross_eval_results_isic_on_pad"

    DATA_DIR_2019 = "data/isic2019"
    DATA_DIR_2020 = "data/isic2020"
    METADATA_2019 = "data/isic2019/ISIC_2019_Training_GroundTruth.csv"
    METADATA_2020 = "data/isic2020/ISIC_2020_Training_GroundTruth_v2.csv"

    IMG_SIZE = 224
    BATCH_SIZE = 256
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create save folder
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # Build an ISIC dataset object to recover the training label ordering
    desired_isic_classes = [
        "melanoma",
        "nevus",
        "basal_cell_carcinoma",
        "seborrheic_keratosis",
    ]
    isic_dataset, isic_split, isic_num_classes, isic_classes, _ = load_isic_dataset(
        DATA_DIR_2019,
        DATA_DIR_2020,
        METADATA_2019,
        METADATA_2020,
        IMG_SIZE,
        desired_classes=desired_isic_classes,
    )

    # Build PAD dataset (target) to get its label order
    pad_dataset, pad_split, pad_num_classes, pad_classes = load_pad_ufes_dataset(
        DATA_DIR_PAD, METADATA_PAD, IMG_SIZE, desired_classes=PAD_DESIRED
    )

    # Build mapping: PAD idx -> ISIC idx (so PAD labels are mapped into the ISIC-trained head order)
    pad_to_isic_idx = build_index_mapping(
        source_labels=list(isic_classes),
        target_labels=list(pad_classes),
        mapping=PAD_TO_ISIC_NAME,
    )

    # Build test dataset and wrap with remapping
    _, _, test_dataset = pad_dataset.get_datasets(pad_split)
    remapped_test_dataset = RemappedLabelsDataset(test_dataset, pad_to_isic_idx)
    test_loader = DataLoader(remapped_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Create model with ISIC num_classes and load ISIC checkpoint
    model = create_efficientnet_model(isic_num_classes).to(DEVICE)
    if not os.path.exists(ISIC_CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {ISIC_CHECKPOINT}")
    
    checkpoint = load_checkpoint(ISIC_CHECKPOINT, model, DEVICE)
    if checkpoint is None:
        raise RuntimeError("Failed to load checkpoint")

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate_model_complete(model, test_loader, criterion, DEVICE, list(isic_classes))

    # Save incorrect predictions
    save_incorrect_predictions_csv(test_metrics['incorrect_predictions'], SAVE_FOLDER)
    
    # Save error analysis
    save_error_analysis_json(test_metrics['error_summary'], SAVE_FOLDER)
    
    # Generate confusion matrix
    y_true = test_metrics['all_labels']
    y_pred = test_metrics['all_predictions']
    
    cm_path = os.path.join(SAVE_FOLDER, 'confusion_matrix.png')
    cm = create_confusion_matrix(y_true, y_pred, list(isic_classes), cm_path)
    
    # Generate classification metrics plot
    metrics_path = os.path.join(SAVE_FOLDER, 'classification_metrics.png')
    create_classification_report_plot(y_true, y_pred, list(isic_classes), metrics_path)
    
    # Generate ROC curves
    y_true_np = np.array(test_metrics['all_labels'])
    outputs_array = np.array(test_metrics['all_outputs'])
    y_pred_proba_np = torch.softmax(torch.tensor(outputs_array), dim=1).numpy()
    roc_path = os.path.join(SAVE_FOLDER, 'roc_curves.png')
    roc_aucs = create_roc_curves(y_true_np, y_pred_proba_np, list(isic_classes), roc_path)
    
    # Create dataset info for saving
    dataset_info = {
        'classes': list(isic_classes),
        'num_classes': isic_num_classes,
        'dataset_sizes': {
            'test': len(test_dataset)
        }
    }

    # Save evaluation results
    save_evaluation_results(test_metrics, dataset_info, SAVE_FOLDER, checkpoint)

    # Print summary
    print_evaluation_summary(test_metrics)
    
    # Print additional metrics
    additional_metrics = test_metrics['additional_metrics']
    print("\n=== MÉTRICAS ADICIONAIS ===")
    print(f"Cohen's Kappa: {additional_metrics['cohen_kappa']:.4f}")
    print(f"Macro Precision: {additional_metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {additional_metrics['macro_recall']:.4f}")
    print(f"Macro F1: {additional_metrics['macro_f1']:.4f}")
    print(f"Weighted Precision: {additional_metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall: {additional_metrics['weighted_recall']:.4f}")
    print(f"Weighted F1: {additional_metrics['weighted_f1']:.4f}")
    print("=" * 35)

    print(f"\nResultados salvos em: {SAVE_FOLDER}")
    print("\n📊 Arquivos gerados:")
    print(f"  - Matriz de confusão: {cm_path}")
    print(f"  - Métricas por classe: {metrics_path}")
    print(f"  - Predições incorretas: {os.path.join(SAVE_FOLDER, 'incorrect_predictions.csv')}")
    print(f"  - Análise de erros: {os.path.join(SAVE_FOLDER, 'error_analysis.json')}")
    print(f"  - Curvas ROC: {roc_path}")
    print(f"  - Resultados completos: {os.path.join(SAVE_FOLDER, 'evaluation_results.json')}")


if __name__ == "__main__":
    # Run both directions by default
    eval_pad_model_on_isic()
    eval_isic_model_on_pad()


