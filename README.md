# Lesion Classifier Generalization

Este repositório tem como objetivo comparar a generalização de modelos de visão treinados para classificação de lesões de pele em datasets distintos.

Serão utilizados modelos baseados em CNNs e ViTs. Os datasets a serem comparados são o PAD-UFES-20 e o ISIC-2020.

## 1. Dependências necessárias

```bash
# Instalar ferramentas de compressão se necessário
sudo apt-get install unzip

# Ou no macOS
brew install unzip
```

## 2. Configuração do ambiente Python com uv

### 2.1. Instalação do uv

O `uv` é um gerenciador de pacotes Python ultra-rápido. Para instalar:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2.2. Configuração do ambiente

```bash
# Criar ambiente virtual e instalar dependências
uv sync

# Ativar o ambiente virtual
source .venv/bin/activate
```

## 3. Preparação do ambiente

### 3.1. Criação da estrutura de diretórios

```bash
mkdir -p data/isic2020
mkdir -p data/pad_ufes_20
```

## 4. Download do Dataset ISIC 2020

### 4.1. Download das imagens e metadados

```bash
cd data/isic2020

# Baixar as imagens de treinamento (JPEG - ~23GB)
curl -C - -L -O --retry 10 --retry-delay 2 --retry-max-time 0 \
  https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip

# Baixar os metadados de treinamento
curl -O https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth_v2.csv

# Baixar as imagens de teste (JPEG - ~6.7GB)
curl -C - -L -O --retry 10 --retry-delay 2 --retry-max-time 0 \
  https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_JPEG.zip

# Baixar os metadados de teste
curl -O https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_Metadata.csv
```

### 4.2. Extração dos arquivos

```bash
# Extrair as imagens de treinamento
unzip ISIC_2020_Training_JPEG.zip

# Extrair as imagens de teste
unzip ISIC_2020_Test_JPEG.zip
```

### 4.3. Verificação dos dados

```bash
# Contar imagens de treinamento
ls ISIC_2020_Training_JPEG/*.jpg | wc -l
# Deve retornar: 33126

# Contar imagens de teste
ls ISIC_2020_Test_JPEG/*.jpg | wc -l
# Deve retornar: 10982
```

### 4.4. Informações sobre o Dataset ISIC 2020

- **Total de imagens de treinamento**: 33,126
- **Total de imagens de teste**: 10,982
- **Formato das imagens**: JPEG
- **Licença**: CC-BY-NC (Creative Commons Attribution-NonCommercial)

#### Distribuição das doenças (Training Set):

**Lesões Benignas (32,542 imagens):**
- **Unknown/Desconhecido**: 27,124 imagens (83.4%)
- **Nevus**: 5,193 imagens (16.0%)
- **Seborrheic Keratosis**: 135 imagens (0.4%)
- **Lentigo NOS**: 44 imagens (0.1%)
- **Lichenoid Keratosis**: 37 imagens (0.1%)
- **Solar Lentigo**: 7 imagens (<0.1%)
- **Cafe-au-lait Macule**: 1 imagem (<0.1%)

**Lesões Malignas (584 imagens):**
- **Melanoma**: 584 imagens (1.8%)
- **Atypical Melanocytic Proliferation**: 1 imagem (<0.1%)

#### Características do dataset:

- **Distribuição**: 98.2% benignas, 1.8% malignas
- **Maioria das imagens**: Marcadas como "unknown" (diagnóstico não especificado)
- **Lesão mais comum**: Nevus (5,193 casos confirmados)
- **Câncer principal**: Melanoma (584 casos)
- **Dados clínicos**: Inclui sexo, idade aproximada, localização anatômica

#### Estrutura dos metadados:

O arquivo CSV contém as seguintes colunas:
- `image_name`: Nome da imagem
- `patient_id`: ID do paciente
- `lesion_id`: ID da lesão
- `sex`: Sexo do paciente (male/female)
- `age_approx`: Idade aproximada
- `anatom_site_general_challenge`: Localização anatômica
- `diagnosis`: Diagnóstico específico
- `benign_malignant`: Classificação binária (benign/malignant)
- `target`: Label numérico (0=benign, 1=malignant)

#### Citação obrigatória

Para usar este dataset, você deve citar:

> International Skin Imaging Collaboration. SIIM-ISIC 2020 Challenge Dataset. International Skin Imaging Collaboration https://doi.org/10.34970/2020-ds01 (2020).

## 5. Download do Dataset PAD-UFES-20

### 5.1. Acesso ao dataset

O PAD-UFES-20 está disponível no Mendeley Data e requer registro para download:

1. Acesse: [https://data.mendeley.com/datasets/zr7vgbcyr2/1](https://data.mendeley.com/datasets/zr7vgbcyr2/1)
2. Clique em "Download All" (requer login/registro)
3. Baixe o arquivo ZIP completo

### 5.2. Download e extração dos dados

```bash
cd data/pad_ufes_20

# Baixar o dataset
curl -C - -L -O --retry 10 --retry-delay 2 --retry-max-time 0 \
  https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip

# Extrair o arquivo baixado
unzip zr7vgbcyr2-1.zip

# Verificar a estrutura dos diretórios
ls -la
```

### 5.3. Verificação dos dados

```bash
# Contar total de imagens
find . -name "*.png" | wc -l
# Deve retornar: 2298

# Verificar metadados
ls *.csv
# Deve mostrar o arquivo de metadados
```

### 5.4. Informações sobre o Dataset PAD-UFES-20

- **Total de amostras**: 2,298
- **Total de pacientes**: 1,373
- **Total de lesões**: 1,641
- **Formato das imagens**: PNG
- **Tipos de lesões**: 6 classes
- **Licença**: CC BY 4.0 (Creative Commons Attribution 4.0)

#### Classes de lesões

1. **Basal Cell Carcinoma (BCC)** - Carcinoma basocelular
2. **Squamous Cell Carcinoma (SCC)** - Carcinoma espinocelular
3. **Melanoma (MEL)** - Melanoma
4. **Actinic Keratosis (ACK)** - Ceratose actínica
5. **Seborrheic Keratosis (SEK)** - Ceratose seborreica
6. **Nevus (NEV)** - Nevo

**Nota**: BCC, SCC e MEL são biópsia-comprovados. Os demais podem ter diagnóstico clínico por consenso de dermatologistas.

#### Características clínicas

Cada amostra inclui até 22 características clínicas:
- Idade do paciente
- Localização da lesão
- Tipo de pele Fitzpatrick
- Diâmetro da lesão
- E outras características clínicas

#### Citação obrigatória

Para usar este dataset, você deve citar:

> Pacheco, Andre G. C., et al. "PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones." Mendeley Data, V1, 2020. [https://doi.org/10.17632/zr7vgbcyr2.1](https://doi.org/10.17632/zr7vgbcyr2.1)

**Artigo relacionado**: [https://doi.org/10.1016/j.compbiomed.2019.103545](https://doi.org/10.1016/j.compbiomed.2019.103545)

## 6. Estrutura final dos dados

Após o download de ambos os datasets, sua estrutura deve ficar assim:

```
data/
├── isic2020/
│   ├── ISIC_2020_Training_JPEG/
│   ├── ISIC_2020_Test_JPEG/
│   ├── ISIC_2020_Training_GroundTruth_v2.csv
│   └── ISIC_2020_Test_Metadata.csv
└── pad_ufes_20/
    ├── images/
    ├── metadata.csv
    └── outros_arquivos/
```


