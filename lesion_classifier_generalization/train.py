"""
Módulo de treinamento para modelos de classificação de lesões
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Training de uma época"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
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
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = total_loss / len(val_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, num_epochs, device, save_folder, 
                learning_rate=1e-5, weight_decay=0.01, patience=5):
    """
    Função principal de treinamento
    
    Args:
        model: Modelo PyTorch
        train_loader: DataLoader para treinamento
        val_loader: DataLoader para validação
        num_epochs: Número de épocas
        device: Dispositivo (CPU/GPU)
        save_folder: Pasta para salvar checkpoints
        learning_rate: Taxa de aprendizado
        weight_decay: Decay do peso
        patience: Paciência para redução de LR
    
    Returns:
        dict: Dicionário com histórico de treinamento
    """
    
    # Loss function e optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.5)
    
    # Log do modelo para wandb
    wandb.watch(model, log="all")
    
    print("Iniciando training...")
    
    best_val_acc = 0.0
    training_history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rate': []
    }
    
    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validation
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
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
        
        # Salvar histórico
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['learning_rate'].append(current_lr)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss:.4f} '
              f'Train Acc: {train_acc:.2f}% '
              f'Val Loss: {val_loss:.4f} '
              f'Val Acc: {val_acc:.2f}% '
              f'LR: {current_lr:.6f}')
        
        # Salvar melhor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(save_folder, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'training_history': training_history
            }, checkpoint_path)
            
            # Log do melhor modelo para wandb
            wandb.run.summary["best_val_accuracy"] = val_acc
            wandb.run.summary["best_epoch"] = epoch
            
            print(f"Novo melhor modelo salvo com val_acc: {val_acc:.2f}%")
    
    print("Training concluído!")
    print(f"Melhor val_acc: {best_val_acc:.2f}%")
    
    return {
        'best_val_acc': best_val_acc,
        'training_history': training_history,
        'checkpoint_path': os.path.join(save_folder, 'best_model.pth')
    }
