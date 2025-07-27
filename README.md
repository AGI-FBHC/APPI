<div align="center">
  <h1>MAPPIS：Multi-channel Attention-enhanced 
  Protein–Protein Interaction Sites predictor</h1>
</div>
<img src="Doc/1.jpg" width="100%">  

## Introduction
(a) Input a protein's 3D structure and its amino acid sequence. Extract sequence information and structural information.    
(b) Construct a directed weighted graph, where each node represents a residue, each edge represents the connection between nodes, and the weight indicates the strength of the connection.  
(c) Apply a channel attention mechanism to assign attention weights to features propagated from different layers (hop distances). The SE-Aggregation (Squeeze-and-Excitation) module is used to aggregate neighbor information across different hops and learn their corresponding importance weights 𝑤1,𝑤2,...,𝑤8, which are then combined through a weighted sum.  
(d) Employ a graph convolutional neural network (GCN) with the incorporation of Initial Residual Connections and Identity Mapping.  
(e) Overall Workflow of MAPPIS  

## Dependency
```markdown
python                    3.10.18
dgl                       2.2.1
freesasa                  2.2.1
matplotlib                3.10.0
numpy                     2.1.2
pandas                    2.3.1
scikit-learn              1.6.1
torch                     2.3.0
torch-cluster             1.6.3
torch-geometric           2.5.0
torch-scatter             2.1.2
torch-sparse              0.6.18
torch-spline-conv         1.2.2
torchaudio                2.3.0
torchdata                 0.8.0
torchvision               0.18.0
```
## Dataset

## Train and Test

### Train
start training
```markdown
python AttPreSite-Ligand_model.py --ligand RNA --trans
```
output
```markdown
./Model/fold1_best_model.pkl
./Model/fold2_best_model.pkl
...
./Model/full_model_30.pkl
```

### Test
