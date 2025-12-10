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

# Variables - Détecter automatiquement le répertoire du projet
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
CODE_DIR="$PROJECT_DIR/Code"
VENV_DIR="$CODE_DIR/envVirtual"

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
    python3 -m venv "$VENV_DIR"
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

# Installer les dépendances uniquement via requirements.txt
if [ -f "$CODE_DIR/requirements.txt" ]; then
    print_info "Installation des dépendances depuis requirements.txt..."
    pip install -r "$CODE_DIR/requirements.txt"
    print_success "Dépendances installées avec succès"
else
    print_error "Fichier requirements.txt non trouvé dans $CODE_DIR. Veuillez le créer et lister toutes les dépendances nécessaires."
    exit 1
fi

# Vérifier que le modèle YOLO existe
if [ ! -f "$CODE_DIR/yolov8n.pt" ]; then
    print_warning "Modèle yolov8n.pt non trouvé. Le téléchargement se fera automatiquement au premier lancement."
fi

# Vérifier les librairies système Qt nécessaires (pour Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    print_info "Vérification des librairies Qt système..."
    if ! ldconfig -p 2>/dev/null | grep -q libxcb; then
        print_warning "libxcb non trouvée. Vous devrez peut-être l'installer :"
        print_warning "  - Ubuntu/Debian: sudo apt install libxcb-xinerama0 python3-tk"
        print_warning "  - Fedora/RHEL: sudo dnf install libxcb qt5-qtbase"
        print_warning "  - Arch/Manjaro: sudo pacman -S qt5-base libxcb"
    fi
fi

# Configurer l'environnement d'affichage pour éviter les problèmes Qt
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb}
    export DISPLAY=${DISPLAY:-:0}
fi

# Lancer le script principal avec le nouveau pipeline
print_info "Lancement de l'interface graphique (UX) :"
cd "$CODE_DIR"
python3 gui_main.py
print_success "Projet terminé avec succès"
