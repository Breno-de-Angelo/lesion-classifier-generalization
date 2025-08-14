import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from raug.loader import MyDataset


class PADUFES20Dataset:
    """
    Classe para carregar e preparar o dataset PAD-UFES-20
    """
    
    def __init__(self, data_dir, metadata_file, img_size=224):
        self.data_dir = data_dir
        self.metadata_file = metadata_file
        self.img_size = img_size
        
        # Carregar metadados
        self.df = pd.read_csv(metadata_file)
        
        # Preparar labels
        self.label_encoder = LabelEncoder()
        self.df['label_encoded'] = self.label_encoder.fit_transform(self.df['diagnostic'])
        
        # Definir transformações
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_image_paths(self):
        """Retorna caminhos das imagens e labels"""
        image_paths = []
        labels = []
        
        for idx, row in self.df.iterrows():
            # Construir caminho da imagem usando img_id (não image_name)
            img_name = row['img_id']
            
            # Procurar a imagem recursivamente nos subdiretórios
            img_path = self._find_image_path(img_name)
            
            if img_path:
                image_paths.append(img_path)
                labels.append(row['label_encoded'])
        
        return image_paths, labels
    
    def _find_image_path(self, img_name):
        """Procura uma imagem recursivamente nos subdiretórios"""
        # Primeiro, tentar no diretório principal
        main_path = os.path.join(self.data_dir, 'images', img_name)
        if os.path.exists(main_path):
            return main_path
        
        # Se não encontrar, procurar recursivamente
        images_dir = os.path.join(self.data_dir, 'images')
        for root, dirs, files in os.walk(images_dir):
            if img_name in files:
                return os.path.join(root, img_name)
        
        return None
    
    def get_metadata_features(self):
        """Retorna features clínicas para treinamento"""
        # Selecionar features numéricas relevantes
        numeric_features = ['age', 'diameter_1', 'diameter_2']
        
        # Preencher valores NaN com médias
        metadata_df = self.df[numeric_features].fillna(self.df[numeric_features].mean())
        
        # Normalizar features
        metadata_df = (metadata_df - metadata_df.mean()) / metadata_df.std()
        
        return metadata_df.values
    
    def split_data(self, test_size=0.2, val_size=0.2, random_state=42):
        """Divide os dados em train/val/test"""
        image_paths, labels = self.get_image_paths()
        metadata = self.get_metadata_features()
        
        # Primeira divisão: train+val vs test
        X_temp, X_test, y_temp, y_test, meta_temp, meta_test = train_test_split(
            image_paths, labels, metadata, test_size=test_size, 
            random_state=random_state, stratify=labels
        )
        
        # Segunda divisão: train vs val
        X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
            X_temp, y_temp, meta_temp, test_size=val_size/(1-test_size), 
            random_state=random_state, stratify=y_temp
        )
        
        return {
            'train': (X_train, y_train, meta_train),
            'val': (X_val, y_val, meta_val),
            'test': (X_test, y_test, meta_test)
        }
    
    def get_datasets(self, split_data):
        """Cria datasets PyTorch"""
        train_dataset = MyDataset(
            imgs_path=split_data['train'][0],
            labels=split_data['train'][1],
            meta_data=split_data['train'][2],
            transform=self.train_transform
        )
        
        val_dataset = MyDataset(
            imgs_path=split_data['val'][0],
            labels=split_data['val'][1],
            meta_data=split_data['val'][2],
            transform=self.val_transform
        )
        
        test_dataset = MyDataset(
            imgs_path=split_data['test'][0],
            labels=split_data['test'][1],
            meta_data=split_data['test'][2],
            transform=self.val_transform
        )
        
        return train_dataset, val_dataset, test_dataset
