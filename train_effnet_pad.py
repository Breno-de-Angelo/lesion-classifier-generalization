"""
Script alternativo para training com Weights & Biases (wandb)
Dataset: PAD-UFES-20
Modelo: EfficientNet
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from lesion_classifier_generalization.pad_ufes_dataset import PADUFES20Dataset


class EfficientNetWithMetadata(nn.Module):
    """
    Modelo EfficientNet com integração de metadados clínicos
    """
    def __init__(self, num_classes, num_metadata_features=3, dropout_rate=0.5):
        super().__init__()
        
        # Carregar EfficientNet pré-treinado
        import torchvision.models as models
        try:
            # Para versões mais recentes do torchvision
            self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        except AttributeError:
            # Fallback para versões mais antigas
            self.efficientnet = models.efficientnet_b0(pretrained=True)
        
        # Remover o classificador final
        self.efficientnet.classifier = nn.Identity()
        
        # Feature reducer (equivalente ao neurons_reducer_block do RAUG)
        self.feature_reducer = nn.Sequential(
            nn.Linear(1280, 512),  # EfficientNet-B0 features
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        
        # Classificador final com metadados
        self.classifier = nn.Linear(512 + num_metadata_features, num_classes)
        
    def forward(self, x, metadata):
        # Extrair features do EfficientNet
        features = self.efficientnet(x)
        
        # Reduzir dimensão das features
        features = self.feature_reducer(features)
        
        # Concatenar com metadados (equivalente ao comb_method='concat' do RAUG)
        combined = torch.cat([features, metadata], dim=1)
        
        # Classificação final
        output = self.classifier(combined)
        
        return output


def create_efficientnet_model(num_classes, num_metadata_features=3):
    """
    Cria modelo EfficientNet com metadados
    """
    model = EfficientNetWithMetadata(
        num_classes=num_classes,
        num_metadata_features=num_metadata_features,
        dropout_rate=0.5
    )
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Training de uma época"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels, metadata, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        metadata = metadata.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Log para wandb a cada 10 batches
        if batch_idx % 10 == 0:
            wandb.log({
                "batch_loss": loss.item(),
                "batch_accuracy": 100. * correct / total
            })
    
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validação de uma época"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, metadata, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            metadata = metadata.to(device).float()
            
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = total_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for images, labels, metadata, _ in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            metadata = metadata.to(device).float()
            
            outputs = model(images, metadata)
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
    
    # Balanced accuracy (simplificado)
    from sklearn.metrics import balanced_accuracy_score
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions) * 100
    
    # AUC (simplificado)
    from sklearn.metrics import roc_auc_score
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


def train_with_wandb():
    """
    Função principal para training com wandb
    """
    
    # Configurações
    DATA_DIR = "data/pad_ufes_20"
    METADATA_FILE = "data/pad_ufes_20/metadata.csv"
    SAVE_FOLDER = "results_pad_ufes_20_wandb"
    IMG_SIZE = 224
    BATCH_SIZE = 16
    NUM_EPOCHS = 2
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Usando dispositivo: {DEVICE}")
    
    # Verificar se os dados existem
    if not os.path.exists(DATA_DIR):
        print(f"Erro: Diretório {DATA_DIR} não encontrado!")
        return
    
    if not os.path.exists(METADATA_FILE):
        print(f"Erro: Arquivo {METADATA_FILE} não encontrado!")
        return
    
    # Criar diretório de resultados
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    
    # Inicializar wandb
    wandb.init(
        project="pad-ufes-20-lesion-classification",
        name="efficientnet-metadata-native",
        config={
            "architecture": "EfficientNet-B0-Native",
            "dataset": "PAD-UFES-20",
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "img_size": IMG_SIZE,
            "device": str(DEVICE)
        }
    )
    
    print("Carregando dataset PAD-UFES-20...")
    
    # Carregar dataset
    dataset = PADUFES20Dataset(DATA_DIR, METADATA_FILE, IMG_SIZE)
    
    # Dividir dados
    print("Dividindo dados em train/val/test...")
    split_data = dataset.split_data(test_size=0.2, val_size=0.2)
    
    # Criar datasets
    train_dataset, val_dataset, test_dataset = dataset.get_datasets(split_data)
    
    print(f"Train: {len(train_dataset)} imagens")
    print(f"Val: {len(val_dataset)} imagens")
    print(f"Test: {len(test_dataset)} imagens")
    
    # Criar dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Criar modelo
    num_classes = len(dataset.label_encoder.classes_)
    num_metadata_features = split_data['train'][2].shape[1]
    
    print(f"Número de classes: {num_classes}")
    print(f"Classes: {dataset.label_encoder.classes_}")
    print(f"Número de features de metadados: {num_metadata_features}")
    
    model = create_efficientnet_model(num_classes, num_metadata_features)
    model = model.to(DEVICE)
    
    # Log do modelo para wandb
    wandb.watch(model, log="all")
    
    # Loss function e optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    print("Iniciando training com wandb...")
    
    best_val_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        
        # Validation
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, DEVICE)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log para wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": current_lr
        })
        
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] '
              f'Train Loss: {train_loss:.4f} '
              f'Train Acc: {train_acc:.2f}% '
              f'Val Loss: {val_loss:.4f} '
              f'Val Acc: {val_acc:.2f}% '
              f'LR: {current_lr:.6f}')
        
        # Salvar melhor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'label_encoder': dataset.label_encoder
            }, os.path.join(SAVE_FOLDER, 'best_model.pth'))
            
            # Log do melhor modelo para wandb
            wandb.run.summary["best_val_accuracy"] = val_acc
            wandb.run.summary["best_epoch"] = epoch
            
            print(f"Novo melhor modelo salvo com val_acc: {val_acc:.2f}%")
    
    print("Training concluído!")
    print(f"Melhor val_acc: {best_val_acc:.2f}%")
    
    # Evaluation no conjunto de teste
    print("Avaliando no conjunto de teste...")
    
    # Carregar melhor modelo
    try:
        # Tentar carregar com weights_only=False para compatibilidade
        checkpoint = torch.load(os.path.join(SAVE_FOLDER, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Modelo carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar checkpoint: {e}")
        print("Usando modelo atual para evaluation...")
    
    # Evaluation usando função nativa
    test_metrics = evaluate_model(model, test_loader, criterion, DEVICE)
    
    # Log dos resultados finais para wandb
    wandb.log({
        "test_loss": test_metrics['loss'],
        "test_accuracy": test_metrics['accuracy'],
        "test_top2_accuracy": test_metrics['top2_acc'],
        "test_balanced_accuracy": test_metrics['balanced_accuracy'],
        "test_auc": test_metrics['auc']
    })
    
    # Atualizar summary do wandb
    wandb.run.summary.update({
        "final_test_accuracy": test_metrics['accuracy'],
        "final_test_balanced_accuracy": test_metrics['balanced_accuracy'],
        "final_test_auc": test_metrics['auc']
    })
    
    print("\n=== RESULTADOS FINAIS ===")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Top-2 Accuracy: {test_metrics['top2_acc']:.4f}")
    print(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    
    # Salvar resultados finais
    import json
    results = {
        'test_metrics': test_metrics,
        'classes': dataset.label_encoder.classes_.tolist(),
        'num_classes': num_classes,
        'num_metadata_features': num_metadata_features,
        'dataset_sizes': {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'test': len(test_dataset)
        },
        'best_val_accuracy': best_val_acc,
        'best_epoch': checkpoint['epoch'] if 'checkpoint' in locals() else epoch
    }
    
    with open(os.path.join(SAVE_FOLDER, 'final_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResultados finais salvos em: {SAVE_FOLDER}/final_results.json")
    print("\n🎉 Training e evaluation concluídos com sucesso!")
    print("Acesse o dashboard do wandb para visualizar todos os resultados!")
    
    # Finalizar wandb
    wandb.finish()


if __name__ == "__main__":
    train_with_wandb()
