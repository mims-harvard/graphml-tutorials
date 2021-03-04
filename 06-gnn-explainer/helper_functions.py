import torch 
import numpy as np
from sklearn import metrics
import time as time
from torch_geometric.data import GraphSAINTRandomWalkSampler
import torch.nn as nn


np.random.seed(274)

def check_masks(mask, y):
    '''Given a split mask, check whether both 0 and 1 are represented
    for every class'''
    labs = y[mask]
    pos_pr = [sum(labs[:,c])/len(labs[:,c]) for c in range(labs.shape[1])]
    if (0 in pos_pr)|(1 in pos_pr):
        return (False)
    else:
        return (True)
    
def make_mask(mask_idx, N):
    mask = torch.zeros(N, dtype=int)
    mask[mask_idx] = 1
    return (mask.bool())


def make_training_split(idx, split_frac, y, num_nodes):
    '''
    Makes a trainig, validation, test split for an idx array.
    Checks through iteration to make sure that the splits have representation of 
    each label (every split has at least 1 node with every value for a class)
    '''
    y = y.numpy()
    #iterate till you get good split masks
    goodmasks = False
    while not(goodmasks):
        #select the indices for proteins with labels
        splitperm = np.random.permutation(idx)
        tr_split = int(np.floor(len(splitperm)*split_frac['train']))
        val_split = int(np.floor(len(splitperm)*split_frac['val']))
        #make the masks
        tr_mask = make_mask(splitperm[:tr_split], num_nodes)
        val_mask = make_mask(splitperm[tr_split:(tr_split+val_split)], num_nodes)
        test_mask = make_mask(splitperm[(tr_split+val_split):], num_nodes)
        #check if the split represents all the class values in each split set. 
        goodmasks = check_masks(tr_mask, y) &  check_masks(val_mask, y) &  check_masks(test_mask, y)

    return ({'Train':tr_mask, 'Val':val_mask, 'Test':test_mask})


#trainer class
class LearnGraphSAINT(): 
    
    ''' Learner Class to train a GNN model. Implements GraphSAINTRandomWalkSampler to generate batches
    Largely adapted from Tutorial 01

    Inputs:
        graph - PyG data object containing graph (must include training masks)
        model - the model to train
        args - hyperparameter class object
    '''

    def __init__(self, graph, model, args, criterion=None):

        self.args = args
        self.graph = graph
        self.model = model.to(self.args.device)

        #build data loader on cpu
        self.loader =  GraphSAINTRandomWalkSampler(graph,
                                                   batch_size=self.args.GraphSAINT['batch_size'],
                                                   walk_length=self.args.GraphSAINT['walk_length'],
                                                   num_steps=self.args.GraphSAINT['num_steps'])
        print ('Data Loader created for learner . . .')

        #Set loss function
        if not criterion: 
            #BCEWithLogitsLoss is a class to use for individual or multiple binary prediction tasks
            criterion = nn.BCEWithLogitsLoss()
        self.criterion = criterion

        #Set optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.w_decay)

        self.train_loss = []
        self.val_loss = []
        self.train_complete = False 


    def learn(self) -> None:
        ''' Trains a model. Reports train and val performance every log_step epoch
        Here the ROC-AUC and Loss is calculated for each batch split and the means are reported'''

        if self.train_complete: return

        #Start timing
        print ('Beginning Training . . .')
        startTime = time.time()

        for epoch in range(self.args.epochs):
            epochs_met = {'train_loss' :[], 'train_auc':[], 'val_loss':[], 'val_auc':[]}
            #loop through batches
            for batch in self.loader: 

                #transfer batch to cuda
                batch = batch.to(self.args.device)

                tl, tauc = self.train_batch(batch)
                self.train_loss.append(tl)
                epochs_met['train_loss'].append(tl)
                epochs_met['train_auc'].append(tauc)

                vl, vauc = self.val_batch(batch)
                self.val_loss.append(vl)
                epochs_met['val_loss'].append(vl)
                epochs_met['val_auc'].append(vauc)


            #report progress
            if epoch % self.args.log_step == 0:
                print('Epoch: {} | Time Elapsed: {:.1f}s \nTrain Loss: {:.4f} | Valid Loss: {:.4f} | Train Auc: {:.4f} | Valid Auc: {:.4f}'.format(
                    epoch, time.time()-startTime,
                    np.mean(epochs_met['train_loss']),
                    np.mean(epochs_met['val_loss']),
                    np.mean(epochs_met['train_auc']),
                    np.mean(epochs_met['val_auc'])))

        self.train_complete = True


    def train_batch(self, batch) -> float:
        ''' Forward pass on a single batch with propogating gradients.
        Returns the loss and ROC-AUC'''
        self.model.train() #important to set when training pytorch models
        labels_batch = batch.y[batch.train_mask]
        self.optim.zero_grad()
        output_batch = self.model.forward(batch.x, batch.edge_index) 
        pred_batch = output_batch[batch.train_mask]
        loss = self.criterion(pred_batch, labels_batch)
        loss.backward()
        self.optim.step()

      #also calculate the roc-auc
        true = labels_batch.detach().cpu()
        pred = pred_batch.detach().cpu()
        auc_c = np.asarray([metrics.roc_auc_score(true[:, c], pred[:, c]) for c in range(pred.shape[1])])
        train_auc = np.mean(auc_c)

        return loss.data.item() , train_auc

    def val_batch(self, batch) -> float:
        ''' Forward pass on a valid split of a single batch.
        Returns the loss and ROC-AUC'''
        self.model.eval()
        labels_batch = batch.y[batch.val_mask]
        output_batch = self.model.forward(batch.x, batch.edge_index) 
        pred_batch = output_batch[batch.val_mask]
        loss = self.criterion(pred_batch, labels_batch)

      #also calculate the roc-auc
        true = labels_batch.detach().cpu()
        pred = pred_batch.detach().cpu()
        auc_c = np.asarray([metrics.roc_auc_score(true[:, c], pred[:, c]) for c in range(pred.shape[1])])
        valid_auc = np.mean(auc_c)

        return loss.data.item() , valid_auc

    def test_cpu(self):
        ''' Gets test performance across the whole graph.
        As graph can be too large to fit in gpu memory, the model is run on
        the cpu. Can take a while for full feed-forward.'''

        #ensure model is already trained
        if not self.train_complete: 
            self.learn()
        self.model.eval()
        #start timing
        startTime = time.time()

        #transfer model to cpu
        mm = self.model.to(torch.device('cpu'))
        labels = self.graph.y[self.graph.test_mask]
        output = mm.forward(graph.x, graph.edge_index) 
        pred = output[self.graph.test_mask]
        #get roc-auc score
        true = labels.detach().cpu()
        pred = pred.detach().cpu()
        auc_c = np.asarray([metrics.roc_auc_score(true[:, c], pred[:, c]) for c in range(pred.shape[1])])

        print ('Model forward on cpu took: {:.2f}'.format(time.time()-startTime))

        return (auc_c)