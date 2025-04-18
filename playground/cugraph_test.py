#pip install cugraph dgl-cuda11.0 torch

import cugraph
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


class GCNLayer(nn.Module):  # <- GNN
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        
    def forward(self, g, feature):
        h = self.linear(feature)  # linear transformation
        degs = g.in_degrees().float().clamp(min=1)  # normalization by node degree
        norm = torch.pow(degs, -0.5).to(h.device).unsqueeze(1)
        h = h * norm
        g.ndata['h'] = h
        g.update_all(dgl.function.u_mul_e('h', 'norm', 'm'),
                    dgl.function.sum('m', 'h'))
        h = g.ndata.pop('h')
        h = h * norm
        return h

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(in_feats, hidden_feats)
        self.layer2 = GCNLayer(hidden_feats, out_feats)
        
    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x


def train(num_nodes, model, dgl_g, loss_fn, optimizer):
    labels = torch.randint(0, 2, (num_nodes,))

    for epoch in range(50):
        model.train()
        logits = model(dgl_g, dgl_g.ndata['feat'])
        loss = loss_fn(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')


def main():
    # pandas df -> cuGraph
    edges = pd.DataFrame({
        'src': [0, 1, 2, 3],
        'dst': [1, 2, 3, 0]
    })
    gdf = cugraph.Graph()
    gdf.from_cudf_edgelist(edges, source='src', destination='dst')

    # cuGraph -> DGL graph
    def cugraph_to_dgl(cugraph_g):
        src, dst = cugraph_g.view_edge_list().to_pandas().values.T  # edge list
        dgl_g = dgl.graph((src, dst), num_nodes=cugraph_g.number_of_vertices())  # DGL graph
        return dgl_g

    dgl_g = cugraph_to_dgl(gdf)

    # node features (-> random)
    num_nodes = dgl_g.number_of_nodes()
    dgl_g.ndata['feat'] = torch.randn(num_nodes, 10)  # 10-dimensional node features

    # create model
    model = GCN(in_feats=10, hidden_feats=16, out_feats=2)  # Assume 2 classes for node classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    train(num_nodes, model, dgl_g, loss_fn, optimizer)


if __name__ == "__main__":
    main()
