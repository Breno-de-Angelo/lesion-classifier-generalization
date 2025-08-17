# Lesion Classifier Generalization

Este repositório tem como objetivo comparar a generalização de modelos de visão treinados para classificação de lesões de pele em datasets distintos.

Serão utilizados modelos baseados em CNNs e ViTs. Os datasets a serem comparados são o PAD-UFES-20 e os datasets ISIC (2019 e 2020 combinados).

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
mkdir -p data/isic2019
mkdir -p data/isic2020
mkdir -p data/pad_ufes_20
```

### 3.2. Download automatizado (Recomendado)

Para facilitar o processo, criamos um script que baixa e extrai todos os datasets automaticamente:

```bash
# Executar o script de download automatizado
./download_datasets.sh
```

**⚠️ Nota**: Este script baixará aproximadamente **40GB** de dados. Certifique-se de ter espaço suficiente e uma conexão estável com a internet.

**Tempo estimado**: 2-4 horas dependendo da velocidade da sua conexão.

### 3.3. Download manual (Alternativa)

Se preferir baixar manualmente ou tiver problemas com o script automatizado, siga as instruções detalhadas nas seções 4 e 5.

## 4. Download dos Datasets ISIC 2019 e 2020

### 4.1. Download do Dataset ISIC 2019

O [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/) foi focado na classificação de imagens dermoscópicas entre nove categorias diagnósticas diferentes.

#### 4.1.1. Download das imagens e metadados

```bash
cd data/isic2019

# Baixar as imagens de treinamento (~9.1GB)
curl -C - -L -O --retry 10 --retry-delay 2 --retry-max-time 0 \
  https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip

# Baixar os metadados de treinamento
curl -O https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv

# Baixar as imagens de teste (~3.6GB)
curl -C - -L -O --retry 10 --retry-delay 2 --retry-max-time 0 \
  https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Test_Input.zip

# Baixar os metadados de teste
curl -O https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Test_GroundTruth.csv
```

#### 4.1.2. Extração dos arquivos

```bash
# Extrair as imagens de treinamento
unzip ISIC_2019_Training_Input.zip

# Extrair as imagens de teste
unzip ISIC_2019_Test_Input.zip
```

#### 4.1.3. Verificação dos dados

```bash
# Contar imagens de treinamento
ls ISIC_2019_Training_Input/*.jpg | wc -l
# Deve retornar: 25331

# Contar imagens de teste
ls ISIC_2019_Test_Input/*.jpg | wc -l
# Deve retornar: 8238
```

### 4.2. Download do Dataset ISIC 2020

O [ISIC 2020 Challenge](https://challenge.isic-archive.com/data/#2020) focou na classificação de lesões de pele com dados clínicos adicionais.

#### 4.2.1. Download das imagens e metadados

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

#### 4.2.2. Extração dos arquivos

```bash
# Extrair as imagens de treinamento
unzip ISIC_2020_Training_JPEG.zip

# Extrair as imagens de teste
unzip ISIC_2020_Test_JPEG.zip
```

#### 4.2.3. Verificação dos dados

```bash
# Contar imagens de treinamento
ls ISIC_2020_Training_JPEG/*.jpg | wc -l
# Deve retornar: 33126

# Contar imagens de teste
ls ISIC_2020_Test_JPEG/*.jpg | wc -l
# Deve retornar: 10982
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

### 4.4. Informações sobre os Datasets ISIC

#### 4.4.1. Dataset ISIC 2019

O [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/) foi focado na classificação de imagens dermoscópicas entre nove categorias diagnósticas diferentes, conforme descrito no desafio oficial.

**Características principais:**
- **Total de imagens de treinamento**: 25,331
- **Total de imagens de teste**: 8,238
- **Formato das imagens**: JPEG
- **Licença**: CC-BY-NC (Creative Commons Attribution-NonCommercial)
- **Métrica de avaliação**: Normalized multi-class accuracy (balanced across categories)

**Nove categorias diagnósticas:**
1. **Melanoma** - Melanoma maligno
2. **Melanocytic nevus** - Nevo melanocítico (benigno)
3. **Basal cell carcinoma** - Carcinoma basocelular
4. **Actinic keratosis** - Ceratose actínica (pré-maligna)
5. **Benign keratosis** - Ceratose benigna (solar lentigo/seborrheic keratosis/lichen planus-like keratosis)
6. **Dermatofibroma** - Dermatofibroma (benigno)
7. **Vascular lesion** - Lesão vascular
8. **Squamous cell carcinoma** - Carcinoma espinocelular
9. **None of the others** - Nenhuma das outras categorias

**Estrutura dos metadados:**
O arquivo CSV contém colunas one-hot encoding para cada categoria diagnóstica, onde cada imagem tem valor 1.0 para sua categoria correta e 0.0 para as demais.

#### 4.4.2. Dataset ISIC 2020

O [ISIC 2020 Challenge](https://challenge.isic-archive.com/data/#2020) focou na classificação de lesões de pele com dados clínicos adicionais e contexto clínico.

**Características principais:**
- **Total de imagens de treinamento**: 33,126
- **Total de imagens de teste**: 10,982
- **Formato das imagens**: JPEG
- **Licença**: CC-BY-NC (Creative Commons Attribution-NonCommercial)

**Distribuição das doenças (Training Set):**

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

**Características do dataset:**
- **Distribuição**: 98.2% benignas, 1.8% malignas
- **Maioria das imagens**: Marcadas como "unknown" (diagnóstico não especificado)
- **Lesão mais comum**: Nevus (5,193 casos confirmados)
- **Câncer principal**: Melanoma (584 casos)
- **Dados clínicos**: Inclui sexo, idade aproximada, localização anatômica

**Estrutura dos metadados:**
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

#### 4.4.3. Citações obrigatórias

**Para o dataset ISIC 2019:**
> International Skin Imaging Collaboration. ISIC 2019 Challenge Dataset. International Skin Imaging Collaboration, 2019.

**Para o dataset ISIC 2020:**
> International Skin Imaging Collaboration. SIIM-ISIC 2020 Challenge Dataset. International Skin Imaging Collaboration https://doi.org/10.34970/2020-ds01 (2020).

**Para o dataset ISIC 2020 (citação completa):**
> Rotemberg, V., Kurtansky, N., Betz-Stablein, B., Caffery, L., Chousakos, E., Codella, N., Combalia, M., Dusza, S., Guitera, P., Gutman, D., Halpern, A., Helba, B., Kittler, H., Kose, K., Langer, S., Lioprys, K., Malvehy, J., Musthaq, S., Nanda, J., Reiter, O., Shih, G., Stratigos, A., Tschandl, P., Weber, J. & Soyer, P. A patient-centric dataset of images and metadata for identifying melanomas using clinical context. Sci Data 8, 34 (2021). https://doi.org/10.1038/s41597-021-00815-z

## 5. Download do Dataset PAD-UFES-20

### 5.1. Download e extração dos dados

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

### 5.2. Verificação dos dados

```bash
# Contar total de imagens
find . -name "*.png" | wc -l
# Deve retornar: 2298

# Verificar metadados
ls *.csv
# Deve mostrar o arquivo de metadados
```

### 5.3. Informações sobre o Dataset PAD-UFES-20

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

Após o download de todos os datasets, sua estrutura deve ficar assim:

```
data/
├── isic2019/
│   ├── ISIC_2019_Training_Input/
│   ├── ISIC_2019_Test_Input/
│   ├── ISIC_2019_Training_GroundTruth.csv
│   └── ISIC_2019_Test_GroundTruth.csv
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
