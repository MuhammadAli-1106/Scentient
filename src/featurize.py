import pandas as pd
import logging
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
from mordred import Calculator, descriptors
   
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def featurize(molecules, output_csv):
    calc = Calculator(descriptors, ignore_3D=True)
    results = []

    for name, mol in molecules:
        try:
            desc_dict = calc(mol).asdict()
            desc_dict['name'] = name
            results.append(desc_dict)
        except Exception as e:
            logging.error(f"Error calculating descriptors for {name}: {e}")

    df = pd.DataFrame(results)
    cols = ['name'] + [c for c in df.columns if c != 'name']  # ensure name first
    df = df[cols]
    df.fillna(0, inplace=True)
    df.to_csv(output_csv, index=False)
    logging.info(f"Saved {len(df)} molecules to {output_csv}")
    