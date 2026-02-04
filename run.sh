#!/bin/bash
# Quick launcher for Keyboard Acoustic Research Tool

echo "========================================"
echo "Keyboard Acoustic Research Tool"
echo "Modular Edition"
echo "========================================"
echo

# Check if virtual environment exists
if [ ! -d "myenv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run: python -m venv myenv"
    echo "Then: source myenv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source myenv/bin/activate

# Check if pygame is installed
python -c "import pygame" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing pygame for smooth audio playback..."
    pip install pygame
fi

# Run the application
echo "Starting application..."
echo
python main.py

# Deactivate when done
deactivate
