"""
Módulo de treinamento para modelos de classificação de lesões
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import time


def format_time(seconds):
    """Formata tempo em segundos para formato legível"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours}h {minutes}m {seconds:.1f}s"


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Training de uma época com indicadores visuais detalhados"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Configurar barra de progresso
    num_batches = len(train_loader)
    pbar = tqdm(
        train_loader, 
        desc=f"Epoch {epoch+1} [TRAIN]", 
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    batch_times = []
    start_time = time.time()
    
    for batch_idx, (images, labels, _) in enumerate(pbar):
        batch_start = time.time()
        
        # Forward pass
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Métricas
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Calcular tempo do batch
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Calcular métricas atuais
        current_loss = total_loss / (batch_idx + 1)
        current_acc = 100. * correct / total
        
        # Atualizar barra de progresso com informações detalhadas
        pbar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Acc': f'{current_acc:.2f}%',
            'Batch': f'{batch_idx+1}/{num_batches}',
            'Time': f'{batch_time:.3f}s'
        })
        
        # Log para wandb a cada 10 batches
        if batch_idx % 10 == 0:
            wandb.log({
                "batch_loss": loss.item(),
                "batch_accuracy": 100. * correct / total,
                "batch_time": batch_time
            })
    
    epoch_time = time.time() - start_time
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    # Estatísticas finais da época
    avg_batch_time = sum(batch_times) / len(batch_times)
    print(f"\n📊 Epoch {epoch+1} [TRAIN] - Resumo:")
    print(f"   ⏱️  Tempo total: {format_time(epoch_time)}")
    print(f"   📈 Loss médio: {epoch_loss:.4f}")
    print(f"   🎯 Acurácia: {epoch_acc:.2f}%")
    print(f"   ⚡ Tempo médio por batch: {avg_batch_time:.3f}s")
    print(f"   📦 Total de batches: {num_batches}")
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device, epoch):
    """Validação de uma época com indicadores visuais"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Configurar barra de progresso para validação
    pbar = tqdm(
        val_loader, 
        desc=f"Epoch {epoch+1} [VAL]", 
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Calcular métricas atuais
            current_loss = total_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            
            # Atualizar barra de progresso
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%',
                'Batch': f'{batch_idx+1}/{len(val_loader)}'
            })
    
    epoch_time = time.time() - start_time
    epoch_loss = total_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    # Estatísticas finais da validação
    print(f"📊 Epoch {epoch+1} [VAL] - Resumo:")
    print(f"   ⏱️  Tempo total: {format_time(epoch_time)}")
    print(f"   📈 Loss médio: {epoch_loss:.4f}")
    print(f"   🎯 Acurácia: {epoch_acc:.2f}%")
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, num_epochs, device, save_folder, 
                learning_rate=1e-5, weight_decay=0.01, patience=5):
    """
    Função principal de treinamento com indicadores visuais aprimorados
    
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
    
    # Informações iniciais
    print("🚀 Iniciando training...")
    print("=" * 80)
    print(f"📊 Configurações:")
    print(f"   🎯 Total de épocas: {num_epochs}")
    print(f"   📚 Dados de treino: {len(train_loader.dataset)} imagens")
    print(f"   🔍 Dados de validação: {len(val_loader.dataset)} imagens")
    print(f"   📦 Batch size: {train_loader.batch_size}")
    print(f"   🧠 Learning rate: {learning_rate}")
    print(f"   ⚖️  Weight decay: {weight_decay}")
    print(f"   💾 Pasta de salvamento: {save_folder}")
    print(f"   🔧 Dispositivo: {device}")
    print("=" * 80)
    
    best_val_acc = 0.0
    training_history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rate': []
    }
    
    # Estatísticas de tempo
    total_start_time = time.time()
    epoch_times = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        print(f"\n{'='*20} EPOCH {epoch+1}/{num_epochs} {'='*20}")
        print(f"⏰ Iniciando em: {time.strftime('%H:%M:%S')}")
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validation
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
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
        
        # Calcular tempo da época
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_remaining = remaining_epochs * avg_epoch_time
        
        # Resumo da época
        print(f"\n🎯 EPOCH {epoch+1} - Resumo Final:")
        print(f"   📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   🔍 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"   🧠 Learning Rate: {current_lr:.6f}")
        if old_lr != current_lr:
            print(f"   ⚡ LR reduzido de {old_lr:.6f} para {current_lr:.6f}")
        
        print(f"   ⏱️  Tempo da época: {format_time(epoch_time)}")
        print(f"   ⏱️  Tempo médio por época: {format_time(avg_epoch_time)}")
        print(f"   ⏱️  Tempo restante estimado: {format_time(estimated_remaining)}")
        
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
            
            print(f"   🏆 🆕 NOVO MELHOR MODELO! Val Acc: {val_acc:.2f}%")
            print(f"   💾 Modelo salvo em: {checkpoint_path}")
        else:
            print(f"   🏆 Melhor val_acc até agora: {best_val_acc:.2f}%")
        
        print(f"{'='*60}")
    
    # Estatísticas finais
    total_time = time.time() - total_start_time
    print(f"\n🎉 TRAINING CONCLUÍDO!")
    print("=" * 80)
    print(f"📊 Estatísticas Finais:")
    print(f"   🏆 Melhor val_acc: {best_val_acc:.2f}%")
    print(f"   ⏱️  Tempo total: {format_time(total_time)}")
    print(f"   ⏱️  Tempo médio por época: {format_time(total_time/num_epochs)}")
    print(f"   📈 Loss final (train/val): {training_history['train_loss'][-1]:.4f}/{training_history['val_loss'][-1]:.4f}")
    print(f"   🎯 Acc final (train/val): {training_history['train_acc'][-1]:.2f}%/{training_history['val_acc'][-1]:.2f}%")
    print(f"   💾 Checkpoint salvo em: {os.path.join(save_folder, 'best_model.pth')}")
    print("=" * 80)
    
    return {
        'best_val_acc': best_val_acc,
        'training_history': training_history,
        'checkpoint_path': os.path.join(save_folder, 'best_model.pth')
    }
