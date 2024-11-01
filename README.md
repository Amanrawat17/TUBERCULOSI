# Drug discovery using Machine learning 
# Molecular Activity Predictor

This Flask application predicts the activity of compounds based on their SMILES representation. It visualizes the molecular structure and computes various molecular properties using RDKit. The application also utilizes a pre-trained stacking classifier to classify the compound as either "Active" or "Inactive."

## Features

- Input a SMILES string to visualize the molecular structure.
- Calculate and display molecular properties such as molecular weight and chemical formula.
- Predict the activity of the molecule using a pre-trained machine learning model.
- User-friendly web interface built with Flask.

## Requirements

To run this application, you will need:

- Python 3.7 or higher
- Flask
- RDKit
- NumPy
- Pickle
- BZ2

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/Amanrawat07/TUBERCULOSI.git

Usage
On the homepage, enter a valid SMILES string in the input field.
Click the submit button to visualize the molecular structure and compute properties.
The application will display:
The molecular image.
The molecular weight.
The chemical formula.
The predicted activity (Active/Inactive) with confidence percentage.
Error Handling
The application includes error handling. If the input SMILES string is invalid or if an error occurs during prediction, an appropriate error message will be displayed.
