import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image



class ISICDataset(Dataset):
    """
    Classe para carregar e preparar os datasets ISIC 2019 e 2020
    """
    
    def __init__(self, data_dir_2019, data_dir_2020, metadata_2019, metadata_2020, 
                 img_size=224, desired_classes=None):
        self.data_dir_2019 = data_dir_2019
        self.data_dir_2020 = data_dir_2020
        self.metadata_2019 = metadata_2019
        self.metadata_2020 = metadata_2020
        self.img_size = img_size
        self.desired_classes = desired_classes
        
        # Carregar metadados
        self.df_2019 = pd.read_csv(metadata_2019)
        self.df_2020 = pd.read_csv(metadata_2020)
        
        # Preparar dados ISIC 2019
        self._prepare_isic_2019()
        
        # Preparar dados ISIC 2020
        self._prepare_isic_2020()
        
        # Combinar datasets de treino
        self._combine_training_datasets()
        
        # Preparar labels
        self.label_encoder = LabelEncoder()
        self.df_combined['label_encoded'] = self.label_encoder.fit_transform(self.df_combined['diagnosis_clean'])
        
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
        
        # Obter caminhos das imagens e labels para treino
        self.image_paths, self.labels, self.patient_ids = self.get_training_image_paths()
        
        # Filtrar dados válidos
        valid_indices = [i for i, path in enumerate(self.image_paths) if path is not None]
        self.image_paths = [self.image_paths[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]
        self.patient_ids = [self.patient_ids[i] for i in valid_indices]
    
    def _prepare_isic_2019(self):
        """Prepara dados do ISIC 2019"""
        print("Preparando dados ISIC 2019...")
        
        # ISIC 2019 tem formato one-hot encoding
        # Encontrar a coluna com valor 1.0 para cada linha
        label_columns = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
        
        # Mapear para nomes mais claros
        label_mapping = {
            'MEL': 'melanoma',
            'NV': 'nevus',
            'BCC': 'basal_cell_carcinoma',
            'AK': 'actinic_keratosis',
            'BKL': 'benign_keratosis',
            'DF': 'dermatofibroma',
            'VASC': 'vascular_lesion',
            'SCC': 'squamous_cell_carcinoma',
            'UNK': 'unknown'
        }
        
        # Encontrar o diagnóstico para cada imagem
        diagnoses = []
        for idx, row in self.df_2019.iterrows():
            for col in label_columns:
                if row[col] == 1.0:
                    diagnoses.append(label_mapping[col])
                    break
            else:
                diagnoses.append('unknown')
        
        self.df_2019['diagnosis_clean'] = diagnoses
        self.df_2019['dataset_source'] = 'ISIC_2019'
        self.df_2019['patient_id'] = self.df_2019['image'].str.split('_').str[0]  # Extrair ID do paciente
        
        print(f"ISIC 2019: {len(self.df_2019)} imagens")
        print(f"Classes encontradas: {self.df_2019['diagnosis_clean'].value_counts().to_dict()}")
    
    def _prepare_isic_2020(self):
        """Prepara dados do ISIC 2020"""
        print("Preparando dados ISIC 2020...")
        
        # ISIC 2020 tem coluna 'diagnosis' direta
        # Limpar e padronizar diagnósticos
        diagnosis_mapping = {
            'nevus': 'nevus',
            'melanoma': 'melanoma',
            'seborrheic keratosis': 'seborrheic_keratosis',
            'lentigo NOS': 'lentigo',
            'lichenoid keratosis': 'lichenoid_keratosis',
            'solar lentigo': 'solar_lentigo',
            'cafe-au-lait macule': 'cafe_au_lait_macule',
            'atypical melanocytic proliferation': 'atypical_melanocytic_proliferation',
            'unknown': 'unknown'
        }
        
        self.df_2020['diagnosis_clean'] = self.df_2020['diagnosis'].map(diagnosis_mapping).fillna('unknown')
        self.df_2020['dataset_source'] = 'ISIC_2020'
        
        print(f"ISIC 2020: {len(self.df_2020)} imagens")
        print(f"Classes encontradas: {self.df_2020['diagnosis_clean'].value_counts().to_dict()}")
    
    def _combine_training_datasets(self):
        """Combina os dois datasets de treino"""
        print("Combinando datasets de treino...")
        
        # Selecionar colunas relevantes e renomear para consistência
        df_2019_clean = self.df_2019[['image', 'diagnosis_clean', 'dataset_source', 'patient_id']].copy()
        df_2019_clean = df_2019_clean.rename(columns={'image': 'image_name'})
        
        df_2020_clean = self.df_2020[['image_name', 'diagnosis_clean', 'dataset_source', 'patient_id']].copy()
        
        # Combinar datasets
        self.df_combined = pd.concat([df_2019_clean, df_2020_clean], ignore_index=True)
        
        # Filtrar por classes desejadas se especificado
        if self.desired_classes:
            self.df_combined = self.df_combined[self.df_combined['diagnosis_clean'].isin(self.desired_classes)]
            print(f"Filtrado para classes: {self.desired_classes}")
        
        print(f"Dataset combinado de treino: {len(self.df_combined)} imagens")
        print(f"Classes finais: {self.df_combined['diagnosis_clean'].value_counts().to_dict()}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Carregar imagem
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Aplicar transformação
        if hasattr(self, 'is_training') and self.is_training:
            image = self.train_transform(image)
        else:
            image = self.val_transform(image)
        
        # Obter label e converter para índice numérico
        label_str = self.labels[idx]
        label_idx = self.label_encoder.transform([label_str])[0]
        
        return image, label_idx, idx
    
    def get_training_image_paths(self):
        """Retorna caminhos das imagens e labels para treino"""
        image_paths = []
        labels = []
        patient_ids = []
        
        for idx, row in self.df_combined.iterrows():
            img_name = row['image_name']
            dataset_source = row['dataset_source']
            
            # Procurar a imagem no diretório apropriado
            img_path = self._find_image_path(img_name, dataset_source)
            
            if img_path:
                image_paths.append(img_path)
                labels.append(row['diagnosis_clean'])
                patient_ids.append(row['patient_id'])
        
        return image_paths, labels, patient_ids
    
    def _find_image_path(self, img_name, dataset_source):
        """Procura uma imagem no diretório apropriado"""
        if dataset_source == 'ISIC_2019':
            # ISIC 2019: procurar em ISIC_2019_Training_Input
            search_dir = os.path.join(self.data_dir_2019, 'ISIC_2019_Training_Input')
            # Adicionar extensão se não tiver
            if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_name = img_name + '.jpg'
        else:
            # ISIC 2020: procurar em train
            search_dir = os.path.join(self.data_dir_2020, 'train')
            # Adicionar extensão se não tiver
            if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_name = img_name + '.jpg'
        
        # Primeiro, tentar no diretório principal
        main_path = os.path.join(search_dir, img_name)
        if os.path.exists(main_path):
            return main_path
        
        # Se não encontrar, procurar recursivamente
        for root, dirs, files in os.walk(search_dir):
            if img_name in files:
                return os.path.join(root, img_name)
        
        return None
    
    def split_data(self, test_size=0.2, val_size=0.2, random_state=42):
        """Divide os dados de treino em train/val/test na proporção 60/20/20"""
        # Filtrar dados válidos
        valid_indices = [i for i, path in enumerate(self.image_paths) if path is not None]
        self.image_paths = [self.image_paths[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]
        self.patient_ids = [self.patient_ids[i] for i in valid_indices]

        print(f"Dividindo {len(self.image_paths)} imagens de treino em train/val/test (60/20/20)...")
        
        # Primeira divisão: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.image_paths, self.labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.labels
        )
        
        # Segunda divisão: train vs val
        # Ajustar val_size para considerar que já separamos test_size
        adjusted_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=adjusted_val_size,
            random_state=random_state,
            stratify=y_temp
        )
        
        print(f"   Imagens - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def get_datasets(self, split_data):
        """Cria datasets PyTorch para treino, validação e teste"""
        # Dataset de treinamento (versão leve)
        train_dataset = ISICDatasetLightweight(
            self, split_data['train'][0], split_data['train'][1], is_training=True
        )
        
        # Dataset de validação (versão leve)
        val_dataset = ISICDatasetLightweight(
            self, split_data['val'][0], split_data['val'][1], is_training=False
        )
        
        # Dataset de teste (versão leve)
        test_dataset = ISICDatasetLightweight(
            self, split_data['test'][0], split_data['test'][1], is_training=False
        )
        
        return train_dataset, val_dataset, test_dataset


class ISICDatasetLightweight(ISICDataset):
    """
    Versão leve do ISICDataset que não refaz a inicialização
    """
    def __init__(self, base_dataset, image_paths, labels, is_training=False):
        # Herdar atributos do dataset base sem refazer inicialização
        self.data_dir_2019 = base_dataset.data_dir_2019
        self.data_dir_2020 = base_dataset.data_dir_2020
        self.metadata_2019 = base_dataset.metadata_2019
        self.metadata_2020 = base_dataset.metadata_2020
        self.img_size = base_dataset.img_size
        self.desired_classes = base_dataset.desired_classes
        
        # Usar o label_encoder do dataset base
        self.label_encoder = base_dataset.label_encoder
        
        # Definir transformações
        self.train_transform = base_dataset.train_transform
        self.val_transform = base_dataset.val_transform
        
        # Definir caminhos e labels específicos para este split
        self.image_paths = image_paths
        self.labels = labels
        self.is_training = is_training
        
        # Não precisamos de patient_ids para datasets leves
        self.patient_ids = []
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Carregar imagem
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Aplicar transformação
        if self.is_training:
            image = self.train_transform(image)
        else:
            image = self.val_transform(image)
        
        # Obter label e converter para índice numérico
        label_str = self.labels[idx]
        label_idx = self.label_encoder.transform([label_str])[0]
        
        return image, label_idx, idx
