# Validate the newly generated molecules usign physico-chemical descriptors 
from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
import numpy as np
import pandas as pd


def physicoChem(data, toCSV = False, **Kwargs):
    
    '''
    
    '''
    
    data['Molecule']= ""
    data["MolecularWeight"] = ""
    data["logP"] = ""
    data["NumberRing"] = ""
    data["NumberAliRing"] = ""
    data["NumberAroRing"] = ""
    data["NumberSatRing"] = ""
    data["NumberRotBond"] = ""
    data["HbondA"] = ""
    data["HbondD"] = ""
    #data["PartialCharge"] = ""
    data["NumberBridgedAtom"] = ""
    data["NumberHeteroAtom"] = ""
    #data["RadGyration"] = ""
    data["TPSA"] = ""
    #data["SolventAccArea"]= ""

    for i in data.index:
        data['Molecule'][i],m=checkMolecule(data['SMILES'][i])
        if(m != None):
            data["MolecularWeight"][i] = Chem.rdMolDescriptors.CalcExactMolWt(m)
            data["logP"][i] = Chem.rdMolDescriptors.CalcCrippenDescriptors(m)[0]
            data["NumberRing"][i] = Chem.rdMolDescriptors.CalcNumRings(m)
            data["NumberAliRing"][i] = Chem.rdMolDescriptors.CalcNumAliphaticRings(m)
            data["NumberAroRing"][i] = Chem.rdMolDescriptors.CalcNumAromaticRings(m)
            data["NumberSatRing"][i] = Chem.rdMolDescriptors.CalcNumSaturatedRings(m)
            data["NumberRotBond"][i] = Chem.rdMolDescriptors.CalcNumRotatableBonds(m)
            data["HbondA"][i] = Chem.rdMolDescriptors.CalcNumHBA(m)
            data["HbondD"][i] = Chem.rdMolDescriptors.CalcNumHBD(m)
            #data["PartialCharge"][i] = Chem.rdMolDescriptors.CalcEEMcharges(m)
            #data["NumberRotBonds"] = Chem.rdMolDescriptors.CalcExactMolWt(m)
            data["NumberBridgedAtom"][i] = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(m)
            data["NumberHeteroAtom"][i] = Chem.rdMolDescriptors.CalcNumHeteroatoms(m)
            #data["RadGyration"][i] = Chem.rdMolDescriptors.CalcRadiusOfGyration(m)
            data["TPSA"][i] = Chem.rdMolDescriptors.CalcTPSA(m)
            #data["SolventAccArea"][i]= Chem.rdFreeSASA.CalcSASA(m)

    if toCSV:
        data.to_csv('HOPVSmiles.csv')
    
    return data

# ----------------------------------------- Extra utility function ---------------------------------------------------------------

def checkMolecule(smi):
    m = Chem.MolFromSmiles(smi)
    #print(m)
    if m is None:
        #print('invalid SMILES')
        err='invalid_SMILES'
        return err,None
    else:
        #print('checking chemistry ...')
        try:
            Chem.SanitizeMol(m)
            #print('valid')
            succ = 'valid'
            return succ,m
        except:
            #print('invalid chemistry')
            err='invalid_chemistry'
            return err,None
