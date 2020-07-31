import numpy as np 
import mathchem as mc 

mols = mc.read_from_sdf('Scripps.sdf')

def save_idx(array, idx_name):
    np.save('%s'%(idx_name), array)

wiener_idx = [mol.wiener_index() for mol in mols]
save_idx(wiener_idx, 'wiener_idx.npy')
print("winer index done!")


hyper_wiener_idx = [mol.hyper_wiener_index() for mol in mols]
save_idx(hyper_wiener_idx, 'hyper_wiener_idx.npy')
print("hyper wiener index done!")

zagreb_idx = [mol.multiplicative_sum_zagreb_index() for mol in mols]
save_idx(zagreb_idx, 'zagreb_idx.npy')
print(len(zagreb_idx))
print("zagreb index done!")
