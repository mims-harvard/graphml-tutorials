from rdkit import Chem
import numpy as np
from pysmiles import read_smiles
import networkx as nx
from molecule import Molecule
import pickle
import pandas as pd



class RegressionData():
    """
    	:param mols: list of nx.Graph molecules describing respective SMILES string
        :param labels: list of labels where each label is a list of three topological indices
        [wiener_idx, hyper_wiener_idx, zagreb_idx]
    """
    def __init__(self, mols, labels):
        self.mols = mols
        self.labels = labels
                    
        self.periodic_table = Chem.GetPeriodicTable()
        self.ams = [nx.to_numpy_matrix(mol, weight='order') for mol in self.mols]
        self.graphs = [nx.from_numpy_matrix(am) for am in self.ams]
        self.element_lists = [mol.nodes(data = 'element') for mol in self.mols]
        
    def create_molecule(self, element_list, label, am):
        """
            :param element_list:  list of integers of atomic number of the molecule 
            :param label: list of three topological indices [wiener_idx, hyper_wiener_idx, zagreb_idx]
            :param am: adjacency matrix of the molecule 
            :return: Molecule object with its attributes specified by above parameters
        """
        nodes = np.array([Chem.rdchem.PeriodicTable.GetAtomicNumber(self.periodic_table, atom[1]) for atom in element_list])
        return Molecule(nodes, label, am)

class ClassificationData():
    """
    	:param file_name: string of file name to be used as property prediction task data
    """
    def __init__(self, file_name):
        self.data = pd.read_csv(file_name)
        
        self.smiles = self.data['smiles']
        self.labels = self.data['activity']
        self.mols = [read_smiles(smile) for smile in self.smiles]
        
        self.periodic_table = Chem.GetPeriodicTable()
        self.ams = [nx.to_numpy_matrix(mol, weight='order') for mol in self.mols]
        self.graphs = [nx.from_numpy_matrix(am) for am in self.ams]
        self.element_lists = [mol.nodes(data = 'element') for mol in self.mols]
        
       
    def create_molecule(self, element_list, label, am):
        """
            :param element_list: list of integers of atomic number of the molecule 
            :param label: if active 1, else 0
            :return: Molecule object with its attributes specified by above parameters
        """
        nodes = np.array([Chem.rdchem.PeriodicTable.GetAtomicNumber(self.periodic_table, atom[1]) for atom in element_list])
        return Molecule(nodes, label, am)
    
    def get_labels(self):
        """
            :return: list of labels of {0,1}
        """
        return self.labels

 
def get_smiles(file_name):
    file = open(file_name, 'r')
    smiles = []
    for i in range(5000):
        line = next(file).strip()
        _,_,smile = line.partition('\t')
        smiles.append(smile)
    return smiles

def save_mols(file_name):
    smiles = get_smiles(file_name)
    mols = [read_smiles(smile) for smile in smiles]
    pickle_out = open("5000_mols.pickle", "wb")
    pickle.dump(mols, pickle_out)
    pickle_out.close()
    
def get_data(data):
    molecules = []
    for i in range (len(data.element_lists)):
        e = data.element_lists[i]
        label = data.labels[i]
        am = data.ams[i]

        mol = data.create_molecule(e, label, am)
        molecules.append(mol)
    
    return molecules

def get_labels(wiener_idx, hyper_wiener_idx, zagreb_idx):
    """
        :param wiener_idx:  np.array of shape [-1, 1] containing wiener index of each molecule 
        :param hyper_wiener_idx: np.array of shape [-1, 1] containing hyper wiener index of each molecule 
        :param zagreb_idx: np.array of shape [-1, 1] containing hyper zagreb index of each molecule 
        :return: np.array of shape [-1, 3] where [wiener_idx, hyper_wiener_idx, zagreb_idx] of each 
        molecule is concatenated
    """
    wiener_idx = np.reshape(wiener_idx, (len(wiener_idx), 1))
    hyper_wiener_idx = np.reshape(hyper_wiener_idx, (len(hyper_wiener_idx), 1))
    zagreb_idx = np.reshape(zagreb_idx, (len(zagreb_idx), 1))
    labels = np.hstack((wiener_idx, hyper_wiener_idx, zagreb_idx))
    labels = np.log10(labels)
    return labels

