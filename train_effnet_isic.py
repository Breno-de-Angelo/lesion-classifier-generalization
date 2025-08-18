"""
Script para training com Weights & Biases (wandb)
Dataset: ISIC 2019 + 2020
Modelo: EfficientNet

Este script usa os módulos organizados do pacote lesion_classifier_generalization
"""

import os
import torch
import torch.nn as nn
import wandb
from lesion_classifier_generalization import (
    create_efficientnet_model,
    load_isic_dataset,
    create_dataloaders,
    get_dataset_info,
    train_model,
    evaluate_model,
    load_checkpoint,
    save_evaluation_results,
    log_evaluation_to_wandb,
    print_evaluation_summary
)


def train_with_wandb():
    """
    Função principal para training com wandb usando módulos organizados
    """

    # Configurações
    DATA_DIR_2019 = "data/isic2019"
    DATA_DIR_2020 = "data/isic2020"
    METADATA_2019 = "data/isic2019/ISIC_2019_Training_GroundTruth.csv"
    METADATA_2020 = "data/isic2020/ISIC_2020_Training_GroundTruth_v2.csv"
    SAVE_FOLDER = "results_isic_2019_2020_wandb"
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 5e-5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Usando dispositivo: {DEVICE}")

    # Criar diretório de resultados
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # Inicializar wandb
    wandb.init(
        project="isic-2019-2020-lesion-classification",
        name="efficientnet-isic",
        config={
            "architecture": "EfficientNet-B3-Native",
            "dataset": "ISIC_2019_2020",
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "img_size": IMG_SIZE,
            "device": str(DEVICE)
        }
    )

    try:
        desired_classes = [
            "melanoma",
            "nevus",
            "basal_cell_carcinoma",
            "seborrheic_keratosis"
        ]
        
        print(f"Classes selecionadas: {desired_classes}")
        
        dataset, split_data, num_classes, classes, dataset_info = load_isic_dataset(
            DATA_DIR_2019, DATA_DIR_2020, 
            METADATA_2019, METADATA_2020, 
            IMG_SIZE, desired_classes=desired_classes
        )
        
        # Criar dataloaders usando módulo organizado
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset, split_data, BATCH_SIZE
        )
        
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

        # Evaluation no conjunto de teste
        print("Avaliando no conjunto de teste...")

        # Carregar melhor modelo
        checkpoint = load_checkpoint(
            training_results['checkpoint_path'], model, DEVICE
        )

        if checkpoint is None:
            print("Usando modelo atual para evaluation...")

        # Loss function para evaluation
        criterion = nn.CrossEntropyLoss()

        # Evaluation usando módulo organizado
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

        print("\n🎉 Training concluído com sucesso!")
        print("Acesse o dashboard do wandb para visualizar todos os resultados!")

    except Exception as e:
        print(f"Erro durante o treinamento: {e}")
        raise
    finally:
        # Finalizar wandb
        wandb.finish()


if __name__ == "__main__":
    train_with_wandb()
