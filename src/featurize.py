import pandas as pd
import logging
from mordred import Calculator, descriptors

def featurize(molecules, output_csv):
    # Initialize Mordred calculator
    calc = Calculator(descriptors, ignore_3D=True)
    results = []
    names = []
    # Loop through molecules
    for name, mol in molecules:
        try:
            desc = calc(mol)
            results.append(desc)
            names.append(name)
        except Exception as e:
            logging.error(f"Error calculating descriptors for {name}: {e}")
    # Convert to DataFrame
    # Add name column
    # Save as CSV
    