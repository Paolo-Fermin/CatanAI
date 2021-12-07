# CatanAI - ECE 4554 Computer Vision Final Project

Settlers of Catan is a multiplayer board game where players barter for limited resources to dominate the fictional island of Catan. It is a game of nearly perfect information, so most information needed to play the game can be determined from a visual inspection of the board. We created a Computer Vision system that analyzes images of Catan boards taken from above. With a combination of traditional algorithms and deep learning methods, our system accurately detects the state of the board from an image and encodes it in a data structure. Specifically, the system constructs a digital model of the board and finds and classifies all settlements, roads, resource tiles, numbers, and seaports within the model. This digital model lays the foundation for a completely autonomous Settlers of Catan system that reads board state from images. 
## Environment setup

This system is meant to work on Python version 3.8 or later. Install all required Python packages using PIP package manager. 

```
pip install -r requirements.txt
```

## Execution 

After setting up the environment, simply execute one file, `main.py` to run the demonstration script. Several matplotlib windows will pop up subsequently. Close out each window to see the next image in the sequence. 

## Authors

This code was developed by Paolo Fermin, Nathan Moeliono, and Sam Schoedel. 