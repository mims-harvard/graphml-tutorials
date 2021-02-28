# GNNExplainer

By Steffan Paul (steffanbpaul@gmail.com)

Explainable AI (XAI) is a growing field of research. As Neural Networks become more prevalant in real-world applications, it is increasingly important that methods are developed to interpret a model's black box to ensure that it is trustworthy and safe for use. Futhermore, XAI methods can be used to learn new insights about our data by interpreting the model's process.

One such method developed for explaining Graph Neural Network models is GNNExplainer ([Ying et al, 2019](https://arxiv.org/abs/1903.03894)). GNNExplainer takes in any trained GNN model and identifies a subgraph around a single node that explains an individual prediction (node, graph or edge prediction). It does this by learning a mask over the adjacency matrix of the node's neighborhood. The optimization of the mask is carried out by maximizing the mutual information between the prediction and the distribution of graph substructures. GNNExplainer also learns a mask over the node feature matrix to highlight important features for the prediction.

In this tutorial, we will walk through how GNNExplainer works using code based on an implementation in [pytorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.GNNExplainer), and implement it to see what the resulting explanations look like. Before we get to GNNExplainer we will briefly walk through the data preparation and model training for a dataset of protein-protein interactions. 

Overall, this tutorial will cover the following:

1. Importing and preparing a dataset for modelling
2. Training a GNN on the data
3. Implementing GNNExplainer
4. Explaining a prediction and visualizing the explanation