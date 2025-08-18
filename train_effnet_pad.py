"""
Script para training com Weights & Biases (wandb)
Dataset: PAD-UFES-20
Modelo: EfficientNet

Este script usa os módulos organizados do pacote lesion_classifier_generalization
"""

import os
import torch
import torch.nn as nn
import wandb
import json
from lesion_classifier_generalization import (
    create_efficientnet_model,
    load_pad_ufes_dataset,
    create_dataloaders,
    get_dataset_info,
    train_model,
    evaluate_model,
    load_checkpoint,
    save_evaluation_results,
    log_evaluation_to_wandb,
    print_evaluation_summary
)


def save_data_split(split_data, save_folder):
    """
    Salva a separação dos dados (train/val/test) para uso posterior na avaliação
    
    Args:
        split_data: Dicionário com dados separados
        save_folder: Pasta para salvar
    """
    # Converter os dados para formato serializável
    serializable_split = {}
    for split_name, (paths, labels) in split_data.items():
        serializable_split[split_name] = {
            'paths': paths,
            'labels': labels
        }
    
    # Salvar em JSON
    split_file = os.path.join(save_folder, 'data_split.json')
    with open(split_file, 'w') as f:
        json.dump(serializable_split, f, indent=2)
    
    print(f"Separação dos dados salva em: {split_file}")
    return split_file


def train_with_wandb():
    """
    Função principal para training com wandb usando módulos organizados
    """

    # Configurações
    DATA_DIR = "data/pad_ufes_20"
    METADATA_FILE = "data/pad_ufes_20/metadata.csv"
    SAVE_FOLDER = "results_pad_ufes_20_wandb"
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Usando dispositivo: {DEVICE}")

    # Criar diretório de resultados
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # Inicializar wandb
    wandb.init(
        project="pad-ufes-20-lesion-classification",
        name="efficientnet-pad",
        config={
            "architecture": "EfficientNet-B3-Native",
            "dataset": "PAD-UFES-20",
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "img_size": IMG_SIZE,
            "device": str(DEVICE)
        }
    )

    try:
        # Carregar dataset usando módulo organizado
        dataset, split_data, num_classes, classes = load_pad_ufes_dataset(
            DATA_DIR, METADATA_FILE, IMG_SIZE, desired_classes=["SEK", "BCC", "NEV", "MEL"]
        )
        
        # Salvar a separação dos dados para uso posterior na avaliação
        print("Salvando separação dos dados...")
        save_data_split(split_data, SAVE_FOLDER)
        
        # Criar dataloaders usando módulo organizado
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset, split_data, BATCH_SIZE
        )
        
        # Obter informações do dataset
        dataset_info = get_dataset_info(dataset, split_data)
        
        # Criar modelo
        model = create_efficientnet_model(num_classes)
        model = model.to(DEVICE)

        # Treinar modelo usando módulo organizado
        training_results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            save_folder=SAVE_FOLDER,
            learning_rate=LEARNING_RATE
        )

        print("Training concluído!")
        print(f"Melhor val_acc: {training_results['best_val_acc']:.2f}%")
        print(f"Melhor checkpoint: {training_results['checkpoint_path']}")


        # Evaluation no conjunto de teste
        print("Avaliando no conjunto de teste...")

        # Carregar melhor modelo
        checkpoint = load_checkpoint(
            training_results['checkpoint_path'], model, DEVICE
        )

        if checkpoint is None:
            print("Usando modelo atual para evaluation...")

        # # Loss function para evaluation
        # criterion = nn.CrossEntropyLoss()

        # # Evaluation usando módulo organizado
        # test_metrics = evaluate_model(model, test_loader, criterion, DEVICE)

        # # Log dos resultados para wandb
        # log_evaluation_to_wandb(test_metrics, dataset_info)

        # # Imprimir resumo da avaliação
        # print_evaluation_summary(test_metrics)

        # # Salvar resultados finais
        # save_evaluation_results(
        #     test_metrics, dataset_info, SAVE_FOLDER, checkpoint
        # )

        print(f"\nResultados finais salvos em: {SAVE_FOLDER}")
        print("📁 Arquivos salvos:")
        print(f"  - Modelo treinado: {training_results['checkpoint_path']}")
        print(f"  - Separação dos dados: {os.path.join(SAVE_FOLDER, 'data_split.json')}")
        # print(f"  - Resultados da avaliação: {os.path.join(SAVE_FOLDER, 'evaluation_results.json')}")
        print("\n🎉 Training e evaluation concluídos com sucesso!")
        print("Acesse o dashboard do wandb para visualizar todos os resultados!")

    except Exception as e:
        print(f"Erro durante o treinamento: {e}")
        raise
    finally:
        # Finalizar wandb
        wandb.finish()


if __name__ == "__main__":
    train_with_wandb()
