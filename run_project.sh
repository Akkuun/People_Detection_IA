#!/bin/bash

# Script pour lancer le projet de détection de personnes avec IA
# Ce script crée automatiquement l'environnement virtuel Python et installe les dépendances

set -e  # Arrêter le script si une commande échoue

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction pour afficher des messages colorés
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Variables
PROJECT_DIR="/home/mathis/Programming/People_Detection_IA"
VENV_DIR="$PROJECT_DIR/venv"
CODE_DIR="$PROJECT_DIR/Code"

print_info "Démarrage du script de lancement du projet People Detection IA"

# Vérifier que Python3 est installé
if ! command -v python3 &> /dev/null; then
    print_error "Python3 n'est pas installé. Veuillez l'installer avant de continuer."
    exit 1
fi

print_info "Python3 trouvé: $(python3 --version)"

# Se déplacer dans le répertoire du projet
cd "$PROJECT_DIR"

# Vérifier si l'environnement virtuel existe déjà
if [ ! -d "$VENV_DIR" ]; then
    print_info "Création de l'environnement virtuel Python..."
    python3 -m venv venv
    print_success "Environnement virtuel créé avec succès"
else
    print_info "Environnement virtuel existant trouvé"
fi

# Activer l'environnement virtuel
print_info "Activation de l'environnement virtuel..."
source "$VENV_DIR/bin/activate"

# Mettre à jour pip
print_info "Mise à jour de pip..."
pip install --upgrade pip

# Installer les dépendances
if [ -f "$CODE_DIR/requirements.txt" ]; then
    print_info "Installation des dépendances depuis requirements.txt..."
    pip install -r "$CODE_DIR/requirements.txt"
    print_success "Dépendances installées avec succès"
else
    print_warning "Fichier requirements.txt non trouvé dans $CODE_DIR"
    print_info "Installation manuelle des dépendances principales..."
    pip install ultralytics opencv-python
fi

# Vérifier que le modèle YOLO existe
if [ ! -f "$CODE_DIR/yolov8n.pt" ]; then
    print_warning "Modèle yolov8n.pt non trouvé. Le téléchargement se fera automatiquement au premier lancement."
fi

# Configurer l'environnement d'affichage pour éviter les problèmes Qt
export QT_QPA_PLATFORM=xcb
export DISPLAY=${DISPLAY:-:0}

# Lancer le projet
print_info "Lancement du projet de détection de personnes..."
print_info "Appuyez sur 'q' dans la fenêtre de la caméra pour arrêter le programme"
print_warning "Si la caméra ne s'ouvre pas, vérifiez qu'aucune autre application ne l'utilise"

cd "$CODE_DIR"
print_info "Lancement avec débogage activé..."
python3 -u testProjet.py 2>&1 | tee /tmp/detection_debug.log
print_info "Log sauvegardé dans /tmp/detection_debug.log"

print_success "Projet terminé avec succès"
