# BioInspired Project 2023/2024

Project for the 'Bio-inspired Artificial Intelligence' course of the Master's Degree in Artificial Intelligence Systems at the University of Trento, A.Y. 2023/2024.

## Description
The project consists of designing and implementing a solution for Optimizing the Bin Packing Problem by using Evolutionary Computations [project paper](Report.pdf).

## Dependencies
The project is written in Python 3.11. The required packages are listed in the [requirements](requirements.txt) file.

## Installation
1. To avoid possible conflicts, it is recommended to use a **virtual environment** to install the required packages. 
    ```
    python3 -m venv bioinspired
    source bioinspired/bin/activate
    ``` 

2. To install the project, clone the repository and install the required packages with the following command:
    ```
    pip install -r requirements.txt
    ```

## How to run the program

The program can be run with the following command:

- 2D Packing Problem - [Mixed-Integer Linear Programming (MIP)]
    ```
    python3 greed_2d.py -W <Container Width> -H <Container Hight> -f <file path> -p <plot> -v <verbose>
    ```
- 2D Packing Problem - [Genetic Algorithm (GA)]
    ```
    python3 greed_2d.py -W <Container Width> -H <Container Hight> -f <file path> -p <plot> -v <verbose>
    -pop <population size> -g <max generations> -n_mut <number of mutations> -p_mut <probability of mutation> -e <number of elites>
    ```
- 3D Packing Problem - [Mixed-Integer Linear Programming (MIP)]
    ```
    python3 greed_3d.py -W <Container Width> -H <Container Hight> -D <Container Depth> -f <file path> -p <plot> -m <interactive plot> -v <verbose>
    ```
- 3D Packing Problem - [Genetic Algorithm (GA)]
    ```
    python3 greed_3d.py -W <Container Width> -H <Container Hight> -D <Container Depth> -f <file path> -p <plot> -m <interactive plot> -v <verbose>
    -pop <population size> -g <max generations> -n_mut <number of mutations> -p_mut <probability of mutation> -e <number of elites>
    ```

  - Help:
      ```
      python3 <program.py> -h
      ```

**Note:** The file with items should be in the folder 'data' and must be a csv format file containing in each row the dimensions of each item.
