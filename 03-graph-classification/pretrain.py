import dgl 
from dgl.data.utils import split_dataset
from dgl.nn.pytorch import SAGEConv, GATConv
import argparse 
import numpy as np 
import networkx as nx
from tqdm import tqdm 
import pickle 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from preprocess import RegressionData, get_data, get_labels

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


hyperparams = {
        "num_features": 119,
        "bsz": 10, 
        "num_epochs": 20,
        "hsz": 500,
        "depth": 5, 
        "num_heads": 5,
        "lr": 5 * (10**-4)}


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        #lift features of the nodes 
        self.lifting_layer = nn.Embedding(hyperparams["num_features"], hyperparams["hsz"])
        
        #latent representations of the nodes         
        self.sageConv1 = SAGEConv(in_feats = hyperparams["hsz"], \
                                                       out_feats = hyperparams["hsz"], \
                                                       aggregator_type = 'lstm')
        
        self.sageConv2 = SAGEConv(in_feats = hyperparams["hsz"], \
                                                       out_feats = hyperparams["hsz"], \
                                                       aggregator_type = 'lstm')
        
        self.sageConv3 = SAGEConv(in_feats = hyperparams["hsz"], \
                                                       out_feats = hyperparams["hsz"], \
                                                       aggregator_type = 'lstm')
        
        self.GAT_conv1 = GATConv(in_feats = hyperparams["hsz"], \
                                 out_feats = hyperparams["hsz"], \
                                 num_heads = hyperparams["num_heads"])
        
        #readout layer (also task specific layer  during pretraining)
        self.output_layer = nn.Linear(hyperparams["hsz"], 3)

    def forward(self, graph):
        x = graph.ndata.pop("molecule_nodes")
        x = self.lifting_layer(x)
        

        x = F.relu(self.sageConv1(graph, x))           
        x = F.relu(self.sageConv2(graph, x))
        x = F.relu(self.sageConv3(graph, x))
        
        x = F.relu(self.GAT_conv1(graph, x))
        x = F.elu(self.avg_attention_heads(x))
        x = self.readout(graph, x)
        
        return x
           
    
    def readout(self, graph, node_feats):
        graph.ndata["latent_vectors"] = self.output_layer(node_feats)
        return dgl.sum_nodes(graph, "latent_vectors")
    
    def avg_attention_heads(self, x):
        return torch.mean(x, dim = 1)


def train(model, train_loader, val_loader):
    model.train()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])

    for epoch in range(hyperparams["num_epochs"]):
        train_loss = []
        for iter, (bg, label) in enumerate(tqdm(train_loader)):

            outputs = model(bg).double()
            label = torch.tensor(label).double()
            loss = loss_fn(outputs,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            
        
        
        val_loss = []
        model = model.eval()
        with torch.no_grad():
            for iter, (bg, label) in enumerate(tqdm(val_loader)):
                outputs = model(bg).double()
                label = torch.tensor(label).double()
                loss = loss_fn(outputs,label)
                val_loss.append(loss.item())
        
        train_loss = np.mean(train_loss)
        val_loss = np.mean(val_loss)
    
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

def build_graph(molecule):
    """
        :param: molecule: Molecule object
        :return: DGL graph of the molecule
    """
    am = molecule.am
    G = nx.from_numpy_matrix(am)
    dgl_G = dgl.DGLGraph()
    dgl_G.from_networkx(G)
    dgl_G.ndata["molecule_nodes"] = molecule.nodes

    return dgl_G

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-f", "--data-file", nargs="*")
    parser.add_argument("-T", "--train", action = "store_true", help = "run training loop")
    parser.add_argument("-t", "--test", action = "store_true", help = "run testing loop")
    parser.add_argument("-s", "--save", action = "store_true", help = "save model.pt")
    parser.add_argument("-l", "--load", action = "store_true", help = "load model.pt")
    
    args = parser.parse_args()
    
    pickle_in = open(args.data_file[0], "rb")
    mols = pickle.load(pickle_in)
    pickle_in.close()
    
    #load numpy files
    wiener_idx = np.load(args.data_file[1])
    hyper_wiener_idx = np.load(args.data_file[2])
    zagreb_idx = np.load(args.data_file[3])
    labels = get_labels(wiener_idx, hyper_wiener_idx, zagreb_idx)
    
    
    data = RegressionData(mols, labels)
    dataset = get_data(data)
    dgl_graphs = [build_graph(data) for data in dataset]
    
    dataset = list(zip(dgl_graphs, labels))
    train_dataset, valid_dataset, test_dataset = split_dataset(dataset, [0.8, 0.1, 0.1], shuffle = True)
    
    def collate(samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels)
    
    train_loader = DataLoader(train_dataset, batch_size=hyperparams["bsz"], shuffle=True,
                         collate_fn=collate)
    valid_loader = DataLoader(valid_dataset, hyperparams["bsz"], shuffle=True,
                         collate_fn=collate)
    test_loader = DataLoader(test_dataset, hyperparams["bsz"], shuffle=True,
                         collate_fn=collate)
    
    model = Model()
      
    if args.load:
        print ("loading saved model...")
        model.load_state_dict(torch.load('./pretrain_model.pt'))
        model_dict = model.state_dict()
        
        
    if args.train:
        print ("running training loop...")
        train(model, train_loader, valid_loader)

    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './pretrain_model.pt')

    
  
    
    