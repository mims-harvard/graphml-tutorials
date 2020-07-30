class Molecule:
    """
    	:param nodes: list of integers of atomic number of the molecule 
        :param label: if task is property prediction task, label is {1,0}
        if task is multi regreession task, label is list of three topological indices, 
        [wiener_idx, hyper_wiener_idx, zagreb_idx]
    """
    
    def __init__(self, nodes, label, am):
        self.nodes = nodes
        self.label = label
        self.am = am
