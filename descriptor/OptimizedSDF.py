#!/usr/bin/env python
# coding: utf-8

# In[1]:


from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
import numpy as np
import os


# In[5]:


dataset = pd.read_excel('../data/Maindatasets.xlsx', sheet_name='main')


# In[6]:


# There might be one or more valid SMILES that can represent one compound
# Thanks to Pat Walters for this information,checkout his excellent blog: https://www.blogger.com/profile/18223198920629617711
def canonical_smiles(smiles):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles] 
    smiles = [Chem.MolToSmiles(mol) for mol in mols]
    return smiles


Canon_SMILES = canonical_smiles(dataset['Smiles'])
# Put the smiles in the dataframe
dataset['canSMILES'] = Canon_SMILES
# Create a list for duplicate smiles
duplicates_smiles = dataset[dataset['canSMILES'].duplicated()]['canSMILES'].values
# Create a list for duplicate smiles
dataset[dataset['canSMILES'].isin(duplicates_smiles)].sort_values(by=['canSMILES'])



# In[13]:


# Calculate all Mordred descriptors for each molecule
def All_Mordred_descriptors(data):

    smiles = data['canSMILES']
    # calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    hmols = [Chem.AddHs(m) for m in mols]
    i = 0

    for mol in hmols:
        mol.SetProp("_Name",data['Coformer'][i])
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        i += 1
    # Chem.MolToMolFile(,  "test.sdf")
    with Chem.SDWriter('optimized3D.sdf') as w:
        for m in hmols:
            w.write(m)
            
    # create all descriptors using sdf file and mordred and writ them in a .csv file "mordred-allcompounds.csv"
    os.system("python -m mordred -3 optimized3D.sdf -o mordred-allcompounds.csv")
    
    return


All_Mordred_descriptors(dataset)

main_mordred = pd.read_csv('mordred-allcompounds.csv')

