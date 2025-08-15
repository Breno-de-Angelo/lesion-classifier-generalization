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


def evaluate_trained_model():
    """
    Função principal para avaliação de modelo treinado
    """

    # Configurações
    DATA_DIR = "data/pad_ufes_20"
    METADATA_FILE = "data/pad_ufes_20/metadata.csv"
    CHECKPOINT_PATH = "results_pad_ufes_20_wandb/best_model.pth"
    SAVE_FOLDER = "evaluation_results"
    IMG_SIZE = 224
    BATCH_SIZE = 256
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Usando dispositivo: {DEVICE}")

    # Verificar se o checkpoint existe
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Erro: Checkpoint {CHECKPOINT_PATH} não encontrado!")
        print("Execute primeiro o script de treinamento ou especifique um checkpoint válido.")
        return

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
            "mode": "evaluation_only"
        }
    )

    try:
        # Carregar dataset usando módulo organizado
        print("Carregando dataset PAD-UFES-20...")
        dataset, split_data, num_classes, classes = load_pad_ufes_dataset(
            DATA_DIR, METADATA_FILE, IMG_SIZE
        )
        
        # Criar dataloaders usando módulo organizado
        print("Criando dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset, split_data, BATCH_SIZE
        )
        
        # Obter informações do dataset
        dataset_info = get_dataset_info(dataset, split_data)
        
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
        test_metrics = evaluate_model(model, test_loader, criterion, DEVICE)

        # Log dos resultados para wandb
        log_evaluation_to_wandb(test_metrics, dataset_info)

        # Imprimir resumo da avaliação
        print_evaluation_summary(test_metrics)

        # Salvar resultados finais
        save_evaluation_results(
            test_metrics, dataset_info, SAVE_FOLDER, checkpoint
        )

        print(f"\nResultados finais salvos em: {SAVE_FOLDER}")
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
