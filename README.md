<div align="center">
  <h1>MAPPIS：Multi-channel Attention-enhanced 
  Protein–Protein Interaction Sites predictor</h1>
</div>
<img src="Doc/1.jpg" width="100%">  

(a) Input a protein's 3D structure and its amino acid sequence. Extract sequence information and structural information.    
(b) Construct a directed weighted graph, where each node represents a residue, each edge represents the connection between nodes, and the weight indicates the strength of the connection.  
(c) Apply a channel attention mechanism to assign attention weights to features propagated from different layers (hop distances). The SE-Aggregation (Squeeze-and-Excitation) module is used to aggregate neighbor information across different hops and learn their corresponding importance weights 𝑤1,𝑤2,...,𝑤8, which are then combined through a weighted sum.
