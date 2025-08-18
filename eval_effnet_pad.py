"""
Script para avaliação de modelo treinado
Dataset: PAD-UFES-20
Modelo: EfficientNet

Este script usa os módulos organizados do pacote lesion_classifier_generalization
para avaliar um modelo já treinado, sem executar treinamento.
"""

import os
import torch
import torch.nn as nn
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import (
    confusion_matrix, classification_report, balanced_accuracy_score, 
    roc_auc_score, precision_recall_fscore_support, cohen_kappa_score
)
from lesion_classifier_generalization import (
    create_efficientnet_model,
    load_pad_ufes_dataset,
    create_dataloaders,
    get_dataset_info,
    evaluate_model,
    load_checkpoint,
    save_evaluation_results,
    log_evaluation_to_wandb,
    print_evaluation_summary
)


def load_data_split(save_folder):
    """
    Carrega a separação dos dados salva durante o treinamento
    
    Args:
        save_folder: Pasta onde está salva a separação dos dados
    
    Returns:
        dict: Dicionário com dados separados ou None se não encontrado
    """
    split_file = os.path.join(save_folder, 'data_split.json')
    
    if not os.path.exists(split_file):
        print(f"Aviso: Arquivo de separação dos dados não encontrado: {split_file}")
        print("Será feita uma nova divisão dos dados (não garantirá consistência com o treinamento)")
        return None
    
    try:
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        print(f"Separação dos dados carregada de: {split_file}")
        return split_data
    except Exception as e:
        print(f"Erro ao carregar separação dos dados: {e}")
        return None


def create_dataloaders_from_split(dataset, split_data, batch_size, num_workers=4):
    """
    Cria dataloaders usando a separação de dados carregada
    
    Args:
        dataset: Dataset PAD-UFES-20
        split_data: Dados separados carregados
        batch_size: Tamanho do batch
        num_workers: Número de workers para carregamento
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    # Criar datasets usando a separação carregada
    train_dataset, val_dataset, test_dataset = dataset.get_datasets_from_split(split_data)
    
    print(f"Train: {len(train_dataset)} imagens")
    print(f"Val: {len(val_dataset)} imagens")
    print(f"Test: {len(test_dataset)} imagens")
    
    # Criar dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Não embaralhar para avaliação
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


def create_confusion_matrix(y_true, y_pred, classes, save_path):
    """
    Cria e salva matriz de confusão
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições
        classes: Lista de classes
        save_path: Caminho para salvar a imagem
    """
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    
    # Normalizar a matriz
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Criar heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão Normalizada')
    plt.xlabel('Predição')
    plt.ylabel('Valor Real')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm


def create_classification_report_plot(y_true, y_pred, classes, save_path):
    """
    Cria e salva gráfico do relatório de classificação
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições
        classes: Lista de classes
        save_path: Caminho para salvar a imagem
    """
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
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Recall
    bars2 = ax2.bar(x_pos, recall, width=bar_width, color='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=1)
    ax2.set_title('Recall por Classe', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # F1-Score
    bars3 = ax3.bar(x_pos, f1, width=bar_width, color='salmon', alpha=0.8, edgecolor='darkred', linewidth=1)
    ax3.set_title('F1-Score por Classe', fontsize=14, fontweight='bold', pad=20)
    ax3.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1.05)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(classes, rotation=45, ha='right')
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


def create_roc_curves(y_true, y_pred_proba, classes, save_path):
    """
    Cria e salva curvas ROC para cada classe
    
    Args:
        y_true: Labels verdadeiros
        y_pred_proba: Probabilidades preditas (n_samples, n_classes)
        classes: Lista de classes
        save_path: Caminho para salvar a imagem
    """
    from sklearn.metrics import roc_curve, auc
    
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
        
        # Plotar curva ROC
        plt.plot(fpr, tpr, lw=2, 
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Plotar linha diagonal (classificador aleatório)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Classificador Aleatório (AUC = 0.500)')
    
    # Configurações do gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
    plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)')
    plt.title('Curvas ROC por Classe - PAD-UFES-20')
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
                        'image_path': image_paths[i] if isinstance(image_paths, list) else str(image_paths[i]),
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
        # Usar numpy.array primeiro para evitar warning de performance
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
        import json
        json.dump(error_summary, f, indent=2)
    
    print(f"Análise de erros salva em: {summary_file}")


def evaluate_trained_model():
    """
    Função principal para avaliação de modelo treinado
    """

    # Configurações
    DATA_DIR = "data/pad_ufes_20"
    METADATA_FILE = "data/pad_ufes_20/metadata.csv"
    CHECKPOINT_PATH = "results_pad_ufes_20_wandb/best_model.pth"
    SAVE_FOLDER = "evaluation_results_pad"
    IMG_SIZE = 224
    BATCH_SIZE = 256
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Usando dispositivo: {DEVICE}")

    # Verificar se o checkpoint existe
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Erro: Checkpoint {CHECKPOINT_PATH} não encontrado!")
        print("Execute primeiro o script de treinamento ou especifique um checkpoint válido.")
        return

    # Verificar se a separação dos dados existe
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    data_split = load_data_split(checkpoint_dir)
    
    if data_split is None:
        print("⚠️  Aviso: Não foi possível carregar a separação dos dados do treinamento.")
        print("Será feita uma nova divisão dos dados, mas isso pode não ser consistente com o treinamento.")

    # Criar diretório de resultados
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # Inicializar wandb para logging
    wandb.init(
        project="pad-ufes-20-lesion-classification",
        name="efficientnet-evaluation",
        config={
            "architecture": "EfficientNet-B3-Native",
            "dataset": "PAD-UFES-20",
            "batch_size": BATCH_SIZE,
            "img_size": IMG_SIZE,
            "device": str(DEVICE),
            "mode": "evaluation_only",
            "using_saved_split": data_split is not None
        }
    )

    try:
        # Carregar dataset usando módulo organizado
        print("Carregando dataset PAD-UFES-20...")
        # Usar as mesmas classes do treinamento para compatibilidade
        dataset, _, num_classes, classes = load_pad_ufes_dataset(
            DATA_DIR, METADATA_FILE, IMG_SIZE, desired_classes=["SEK", "BCC", "NEV", "MEL"]
        )
        
        # Criar dataloaders usando a separação salva ou nova divisão
        print("Criando dataloaders...")
        if data_split is not None:
            # Usar separação salva durante o treinamento
            print("✅ Usando separação dos dados salva durante o treinamento")
            train_loader, val_loader, test_loader = create_dataloaders_from_split(
                dataset, data_split, BATCH_SIZE
            )
        else:
            # Fazer nova divisão (não garantirá consistência)
            print("⚠️  Fazendo nova divisão dos dados (não garantirá consistência com treinamento)")
            _, split_data, _, _ = load_pad_ufes_dataset(
                DATA_DIR, METADATA_FILE, IMG_SIZE, desired_classes=["SEK", "BCC", "NEV", "MEL"]
            )
            train_loader, val_loader, test_loader = create_dataloaders(
                dataset, split_data, BATCH_SIZE
            )
        
        # Obter informações do dataset
        if data_split is not None:
            # Usar a separação carregada para obter informações corretas
            dataset_info = {
                'classes': classes,
                'num_classes': num_classes,
                'dataset_sizes': {
                    'train': len(data_split['train']['paths']),
                    'val': len(data_split['val']['paths']),
                    'test': len(data_split['test']['paths'])
                }
            }
        else:
            # Usar informações básicas do dataset
            dataset_info = {
                'classes': classes,
                'num_classes': num_classes,
                'dataset_sizes': {
                    'train': len(dataset.image_paths),
                    'val': 0,
                    'test': 0
                }
            }
        
        print(f"Número de classes: {num_classes}")
        print(f"Classes: {classes}")

        # Criar modelo
        print("Criando modelo...")
        model = create_efficientnet_model(num_classes)
        model = model.to(DEVICE)

        # Carregar checkpoint treinado
        print(f"Carregando checkpoint: {CHECKPOINT_PATH}")
        checkpoint = load_checkpoint(CHECKPOINT_PATH, model, DEVICE)

        if checkpoint is None:
            print("Erro: Não foi possível carregar o checkpoint!")
            return

        print(f"Checkpoint carregado com sucesso!")
        print(f"Época: {checkpoint.get('epoch', 'N/A')}")
        print(f"Validação Accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")

        # Loss function para evaluation
        criterion = nn.CrossEntropyLoss()

        # Evaluation no conjunto de teste
        print("\nAvaliando no conjunto de teste...")
        test_metrics = evaluate_model_complete(model, test_loader, criterion, DEVICE, classes)

        # Salvar predições incorretas
        print("\nSalvando predições incorretas...")
        save_incorrect_predictions_csv(test_metrics['incorrect_predictions'], SAVE_FOLDER)
        
        # Salvar análise de erros
        save_error_analysis_json(test_metrics['error_summary'], SAVE_FOLDER)
        
        # Gerar matriz de confusão usando resultados já calculados
        print("\nGerando matriz de confusão...")
        y_true = test_metrics['all_labels']
        y_pred = test_metrics['all_predictions']
        
        # Criar e salvar matriz de confusão
        cm_path = os.path.join(SAVE_FOLDER, 'confusion_matrix.png')
        cm = create_confusion_matrix(y_true, y_pred, classes, cm_path)
        
        # Criar e salvar gráfico de métricas por classe
        metrics_path = os.path.join(SAVE_FOLDER, 'classification_metrics.png')
        create_classification_report_plot(y_true, y_pred, classes, metrics_path)
        
        # Gerar e salvar curvas ROC
        print("\nGerando curvas ROC...")
        y_true_np = np.array(y_true)
        # Converter outputs para probabilidades usando softmax
        outputs_array = np.array(test_metrics['all_outputs'])
        y_pred_proba_np = torch.softmax(torch.tensor(outputs_array), dim=1).numpy()
        roc_curves_path = os.path.join(SAVE_FOLDER, 'roc_curves.png')
        roc_aucs = create_roc_curves(y_true_np, y_pred_proba_np, classes, roc_curves_path)
        
        # Log dos resultados para wandb
        log_evaluation_to_wandb(test_metrics, dataset_info)
        
        # Log da matriz de confusão para wandb
        wandb.log({
            "confusion_matrix": wandb.Image(cm_path),
            "classification_metrics": wandb.Image(metrics_path),
            "roc_curves": wandb.Image(roc_curves_path)
        })

        # Imprimir resumo da avaliação
        print_evaluation_summary(test_metrics)
        
        # Imprimir métricas adicionais
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

        # Salvar resultados finais
        save_evaluation_results(
            test_metrics, dataset_info, SAVE_FOLDER, checkpoint
        )
        
        print(f"\nResultados finais salvos em: {SAVE_FOLDER}")
        print("\n📊 Arquivos gerados:")
        print(f"  - Matriz de confusão: {cm_path}")
        print(f"  - Métricas por classe: {metrics_path}")
        print(f"  - Predições incorretas: {os.path.join(SAVE_FOLDER, 'incorrect_predictions.csv')}")
        print(f"  - Análise de erros: {os.path.join(SAVE_FOLDER, 'error_analysis.json')}")
        print(f"  - Curvas ROC: {roc_curves_path}")
        
        if data_split is not None:
            print("\n✅ Avaliação realizada usando os mesmos dados de teste do treinamento!")
        else:
            print("\n⚠️  Avaliação realizada com nova divisão dos dados (não garantida consistência com treinamento)")
        
        print("\n🎉 Avaliação concluída com sucesso!")
        print("Acesse o dashboard do wandb para visualizar todos os resultados!")

    except Exception as e:
        print(f"Erro durante a avaliação: {e}")
        raise
    finally:
        # Finalizar wandb
        wandb.finish()


if __name__ == "__main__":
    evaluate_trained_model()
