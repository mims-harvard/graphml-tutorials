import dgl 
from dgl.data.utils import split_dataset
from dgl.nn.pytorch import SAGEConv, GATConv
import argparse 
import numpy as np 
import networkx as nx
from tqdm import tqdm 
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from preprocess import ClassificationData, get_data

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
        ###########shared layers###########
        self.lifting_layer = nn.Embedding(hyperparams["num_features"], hyperparams["hsz"])
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
        ###################################
        
        #######task specific layer(s)#######
        self.readout_layer = nn.Linear(hyperparams["hsz"], 1)
        ###################################
   
    
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
        graph.ndata["latent_vectors"] = self.readout_layer(node_feats)
        return dgl.sum_nodes(graph, "latent_vectors")
    

    def roc_auc(self, logits, labels):

        y_scores = torch.sigmoid(logits)
        y_true = labels
        
        roc_auc = roc_auc_score(y_true, y_scores.detach().numpy())

        print("ROC AUC: ", roc_auc)

        return roc_auc
    
    def avg_attention_heads(self, x):
        return torch.mean(x, dim = 1)
    
    def concat_attention_heads(self, x):
        s = x.size()
        bsz, num_heads, hsz = s[0], s[1], s[2]
        return torch.reshape(x, (bsz, num_heads * hsz))
    
      
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

def finetune(model, train_loader, val_loader):
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
        

   
def test(model, data_loader):
    logits = []
    labels = []
    with torch.no_grad():
        model.eval()
        
        for iter, (bg, label) in enumerate(tqdm(train_loader)):
            logits.append(torch.squeeze(model(bg)))
            labels.append(torch.tensor(label).double())
        logits = torch.cat(logits)
        labels = torch.cat(labels)
        roc_auc, recall, precision = model.roc_auc(logits, labels)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-f", "--data-file", nargs="*")
    parser.add_argument("-F", "--finetune", action = "store_true", help = "run training loop")
    parser.add_argument("-t", "--test", action = "store_true", help = "run testing loop")
    parser.add_argument("-s", "--save", action = "store_true", help = "save model.pt")
    parser.add_argument("-l", "--load", action = "store_true", help = "load model.pt")
    parser.add_argument("-p", "--load-pretrain", action = "store_true", help = "load model.pt")
    
    args = parser.parse_args()
        
    data = ClassificationData(args.data_file[0])
    dataset = get_data(data)
    dgl_graphs = [build_graph(data) for data in dataset]
    labels = data.get_labels() 
    
    dataset = list(zip(dgl_graphs, labels))
    
    train_dataset, valid_dataset, test_dataset = split_dataset(dataset, [0.8, 0.1, 0.1], shuffle = True)
    
    def collate(samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels)
    
    
    train_loader = DataLoader(train_dataset, hyperparams["bsz"], shuffle=True,
                         collate_fn=collate)
    valid_loader = DataLoader(valid_dataset, hyperparams["bsz"], shuffle=True,
                         collate_fn=collate)
    test_loader = DataLoader(test_dataset, hyperparams["bsz"], shuffle=True,
                         collate_fn=collate)
    
    model = Model()
    
    if args.load_pretrain:
        print ("loading pretrained model...")
        model_dict = model.state_dict()
        pretrained_dict = torch.load('./pretrain_model.pt')
        filtered_pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(filtered_pretrained_dict)
        model.load_state_dict(filtered_pretrained_dict)
        
        print ("adding task specific layer(s)...")
        model.add_module("readout_layer", nn.Linear(hyperparams["hsz"], 1))
        print ("done loading!")
        
        
    if args.load:
        print ("loading saved model...")
        model_dict = model.state_dict()
        finetuned_dict = torch.load('./finetuned_model.pt')
        model.add_module("readout_layer", nn.Linear(hyperparams["hsz"], 1))
        model.load_state_dict(finetuned_dict)


    if args.finetune:
        print ("running finetuning loop...")
        finetune(model, train_loader, valid_loader)

    if args.test:
        print ("running testing loop...")
        test(model, train_loader)
        test(model, test_loader)
    
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './finetuned_model.pt')

    
