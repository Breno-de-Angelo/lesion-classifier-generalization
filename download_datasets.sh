#!/bin/bash

# Script para download automatizado dos datasets ISIC 2019, ISIC 2020 e PAD-UFES-20
# Autor: Lesion Classifier Team
# Data: 2024

set -e  # Parar em caso de erro

echo "🚀 Iniciando download dos datasets de classificação de lesões de pele..."
echo "=" * 60

# Criar diretórios de dados
echo "📁 Criando estrutura de diretórios..."
mkdir -p data/isic2019
mkdir -p data/isic2020
mkdir -p data/pad_ufes_20

# Função para verificar se o download foi bem-sucedido
check_download() {
    local file_path="$1"
    local expected_size="$2"
    
    if [ -f "$file_path" ]; then
        local actual_size=$(stat -c%s "$file_path" 2>/dev/null || stat -f%z "$file_path" 2>/dev/null)
        if [ "$actual_size" -gt "$expected_size" ]; then
            echo "✅ $file_path baixado com sucesso"
            return 0
        else
            echo "❌ $file_path parece estar corrompido (tamanho: $actual_size bytes)"
            return 1
        fi
    else
        echo "❌ $file_path não foi encontrado"
        return 1
    fi
}

# Função para extrair arquivos
extract_file() {
    local file_path="$1"
    local extract_dir="$2"
    
    echo "📦 Extraindo $file_path..."
    if unzip -q "$file_path" -d "$extract_dir"; then
        echo "✅ Extração concluída"
    else
        echo "❌ Erro na extração de $file_path"
        return 1
    fi
}

# ============================================================================
# DOWNLOAD ISIC 2019
# ============================================================================
echo ""
echo "🔬 Baixando Dataset ISIC 2019..."
echo "-" * 40

cd data/isic2019

# URLs do ISIC 2019
ISIC_2019_TRAINING_URL="https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
ISIC_2019_TEST_URL="https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Test_Input.zip"
ISIC_2019_TRAINING_GT_URL="https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"
ISIC_2019_TEST_GT_URL="https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Test_GroundTruth.csv"

# Download das imagens de treinamento (~9.1GB)
echo "📥 Baixando imagens de treinamento ISIC 2019 (~9.1GB)..."
curl -C - -L -O --retry 10 --retry-delay 2 --retry-max-time 0 "$ISIC_2019_TRAINING_URL"

# Download das imagens de teste (~3.6GB)
echo "📥 Baixando imagens de teste ISIC 2019 (~3.6GB)..."
curl -C - -L -O --retry 10 --retry-delay 2 --retry-max-time 0 "$ISIC_2019_TEST_URL"

# Download dos metadados
echo "📥 Baixando metadados ISIC 2019..."
curl -O "$ISIC_2019_TRAINING_GT_URL"
curl -O "$ISIC_2019_TEST_GT_URL"

# Verificar downloads
echo "🔍 Verificando downloads ISIC 2019..."
check_download "ISIC_2019_Training_Input.zip" 9000000000  # ~9GB
check_download "ISIC_2019_Test_Input.zip" 3500000000      # ~3.5GB
check_download "ISIC_2019_Training_GroundTruth.csv" 1000000
check_download "ISIC_2019_Test_GroundTruth.csv" 100000

# Extrair arquivos
echo "📦 Extraindo arquivos ISIC 2019..."
extract_file "ISIC_2019_Training_Input.zip" "."
extract_file "ISIC_2019_Test_Input.zip" "."

# Verificar extração
echo "🔍 Verificando extração ISIC 2019..."
TRAINING_COUNT=$(ls ISIC_2019_Training_Input/*.jpg 2>/dev/null | wc -l)
TEST_COUNT=$(ls ISIC_2019_Test_Input/*.jpg 2>/dev/null | wc -l)

echo "📊 ISIC 2019 - Imagens de treinamento: $TRAINING_COUNT"
echo "📊 ISIC 2019 - Imagens de teste: $TEST_COUNT"

cd ..

# ============================================================================
# DOWNLOAD ISIC 2020
# ============================================================================
echo ""
echo "🔬 Baixando Dataset ISIC 2020..."
echo "-" * 40

cd isic2020

# URLs do ISIC 2020
ISIC_2020_TRAINING_URL="https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip"
ISIC_2020_TEST_URL="https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_JPEG.zip"
ISIC_2020_TRAINING_GT_URL="https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth_v2.csv"
ISIC_2020_TEST_METADATA_URL="https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_Metadata.csv"

# Download das imagens de treinamento (~23GB)
echo "📥 Baixando imagens de treinamento ISIC 2020 (~23GB)..."
curl -C - -L -O --retry 10 --retry-delay 2 --retry-max-time 0 "$ISIC_2020_TRAINING_URL"

# Download das imagens de teste (~6.7GB)
echo "📥 Baixando imagens de teste ISIC 2020 (~6.7GB)..."
curl -C - -L -O --retry 10 --retry-delay 2 --retry-max-time 0 "$ISIC_2020_TEST_URL"

# Download dos metadados
echo "📥 Baixando metadados ISIC 2020..."
curl -O "$ISIC_2020_TRAINING_GT_URL"
curl -O "$ISIC_2020_TEST_METADATA_URL"

# Verificar downloads
echo "🔍 Verificando downloads ISIC 2020..."
check_download "ISIC_2020_Training_JPEG.zip" 22000000000  # ~22GB
check_download "ISIC_2020_Test_JPEG.zip" 6000000000       # ~6GB
check_download "ISIC_2020_Training_GroundTruth_v2.csv" 2000000
check_download "ISIC_2020_Test_Metadata.csv" 400000

# Extrair arquivos
echo "📦 Extraindo arquivos ISIC 2020..."
extract_file "ISIC_2020_Training_JPEG.zip" "."
extract_file "ISIC_2020_Test_JPEG.zip" "."

# Verificar extração
echo "🔍 Verificando extração ISIC 2020..."
TRAINING_COUNT=$(ls ISIC_2020_Training_JPEG/*.jpg 2>/dev/null | wc -l)
TEST_COUNT=$(ls ISIC_2020_Test_JPEG/*.jpg 2>/dev/null | wc -l)

echo "📊 ISIC 2020 - Imagens de treinamento: $TRAINING_COUNT"
echo "📊 ISIC 2020 - Imagens de teste: $TEST_COUNT"

cd ..

# ============================================================================
# DOWNLOAD PAD-UFES-20
# ============================================================================
echo ""
echo "🔬 Baixando Dataset PAD-UFES-20..."
echo "-" * 40

cd pad_ufes_20

# URL do PAD-UFES-20
PAD_UFES_URL="https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip"

# Download do dataset
echo "📥 Baixando PAD-UFES-20..."
curl -C - -L -O --retry 10 --retry-delay 2 --retry-max-time 0 "$PAD_UFES_URL"

# Verificar download
echo "🔍 Verificando download PAD-UFES-20..."
check_download "zr7vgbcyr2-1.zip" 100000000  # ~100MB

# Extrair arquivo
echo "📦 Extraindo PAD-UFES-20..."
extract_file "zr7vgbcyr2-1.zip" "."

# Verificar extração
echo "🔍 Verificando extração PAD-UFES-20..."
TOTAL_IMAGES=$(find . -name "*.png" 2>/dev/null | wc -l)
METADATA_FILES=$(ls *.csv 2>/dev/null | wc -l)

echo "📊 PAD-UFES-20 - Total de imagens: $TOTAL_IMAGES"
echo "📊 PAD-UFES-20 - Arquivos de metadados: $METADATA_FILES"

cd ..

# ============================================================================
# RESUMO FINAL
# ============================================================================
echo ""
echo "🎉 Download e extração concluídos!"
echo "=" * 60
echo "📊 Resumo dos datasets baixados:"
echo ""
echo "🔬 ISIC 2019:"
echo "   • Treinamento: $(ls data/isic2019/ISIC_2019_Training_Input/*.jpg 2>/dev/null | wc -l) imagens"
echo "   • Teste: $(ls data/isic2019/ISIC_2019_Test_Input/*.jpg 2>/dev/null | wc -l) imagens"
echo ""
echo "🔬 ISIC 2020:"
echo "   • Treinamento: $(ls data/isic2020/ISIC_2020_Training_JPEG/*.jpg 2>/dev/null | wc -l) imagens"
echo "   • Teste: $(ls data/isic2020/ISIC_2020_Test_JPEG/*.jpg 2>/dev/null | wc -l) imagens"
echo ""
echo "🔬 PAD-UFES-20:"
echo "   • Total: $(find data/pad_ufes_20 -name "*.png" 2>/dev/null | wc -l) imagens"
echo ""
echo "📁 Estrutura criada em: data/"
echo "💡 Use os scripts de treinamento para começar:"
echo "   • python train_effnet_pad.py     # Para PAD-UFES-20"
echo "   • python train_effnet_isic.py    # Para ISIC 2019+2020"
echo "   • python demo_isic_classes.py    # Para explorar classes ISIC"
echo ""
echo "🚀 Pronto para treinar modelos de classificação de lesões!"
