---
title: "Graph Neural Networks in Practice: Social Networks, Drug Discovery, and Beyond"
meta_title: ""
description: "A comprehensive exploration of Graph Neural Networks and their transformative applications in social network analysis, drug discovery, and other domains where relational data reigns supreme."
date: 2028-12-14
image: "/images/blogs/gnns-practice/cover.jpg"
categories: ["Machine Learning"]
author: "Amar Singh"
tags: ["gnn", "graph-neural-networks", "drug-discovery", "social-networks"]
draft: false
---

Graph Neural Networks (GNNs) represent one of the most exciting frontiers in modern machine learning. While convolutional neural networks revolutionized image processing and transformers transformed natural language understanding, GNNs have quietly become the architecture of choice for a vast category of problems that neither CNNs nor transformers were designed to handle: data that lives on graphs. From predicting how molecules will interact with proteins to detecting fraudulent transactions in financial networks, GNNs operate on the fundamental insight that relationships between entities are often just as important as the entities themselves.

In this post, we will explore GNNs from the ground up, understand the mathematical intuition behind message passing, and then dive deep into two of the most impactful application domains: social network analysis and drug discovery. Along the way, we will examine real code patterns, discuss architectural choices, and explore the challenges that practitioners face when deploying GNNs in production.

## Why Graphs Matter

Most of the data we encounter in the real world has an inherent relational structure. A social network is a graph where users are nodes and friendships are edges. A molecule is a graph where atoms are nodes and chemical bonds are edges. A knowledge graph connects entities through typed relationships. A transportation network links locations through routes. Even images and text can be represented as graphs, though specialized architectures tend to work better for those modalities.

Traditional machine learning approaches struggle with graph-structured data for several reasons. First, graphs have no fixed size or shape. Unlike images, which can be resized to a standard dimension, or text, which can be tokenized into sequences, graphs vary wildly in their topology. Second, graphs have no canonical ordering of their nodes. You cannot simply flatten a graph into a vector without losing structural information. Third, the important information in a graph is often encoded in the connectivity pattern itself, not just in the features of individual nodes.

GNNs address these challenges by operating directly on the graph structure, learning representations that capture both node features and the topology of their local neighborhoods.

## The Message Passing Framework

At their core, most GNNs follow a message passing framework. The idea is elegantly simple: each node updates its representation by aggregating information from its neighbors. After multiple rounds of message passing, each node's representation encodes information about its extended neighborhood in the graph.

Formally, a single layer of message passing can be described in three steps. First, for each edge in the graph, a message is computed based on the features of the source node, the target node, and optionally the edge features. Second, for each node, all incoming messages are aggregated using a permutation-invariant function such as sum, mean, or max. Third, each node's representation is updated by combining its current features with the aggregated messages, typically through a neural network layer.

Mathematically, the update for node v at layer k can be written as:

```
h_v^(k) = UPDATE(h_v^(k-1), AGGREGATE({h_u^(k-1) : u in N(v)}))
```

where N(v) denotes the neighbors of node v, AGGREGATE is a permutation-invariant function, and UPDATE is typically a learned transformation.

This framework is powerful because it naturally handles graphs of arbitrary size and structure. The same learned parameters are applied at every node, regardless of its degree or position in the graph. After K layers of message passing, each node's representation captures information from its K-hop neighborhood.

## Key GNN Architectures

Several important GNN architectures have been proposed, each implementing the message passing framework with different design choices.

### Graph Convolutional Networks (GCN)

The Graph Convolutional Network, introduced by Kipf and Welling in 2017, is perhaps the most widely known GNN architecture. GCN uses a simple aggregation scheme where each node's representation is updated by taking a weighted average of its neighbors' features, followed by a linear transformation and nonlinearity. The weights are determined by the graph structure through a normalized adjacency matrix.

The GCN layer can be written as:

```python
# Simplified GCN layer in PyTorch Geometric
import torch
from torch_geometric.nn import GCNConv

class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

GCN's simplicity is both its strength and its limitation. It works well for homophilous graphs where connected nodes tend to have similar labels, but it can struggle with heterophilous graphs where neighbors are more likely to belong to different classes.

### GraphSAGE

GraphSAGE (Sample and Aggregate), introduced by Hamilton, Ying, and Leskovec in 2017, addresses two key limitations of GCN. First, it samples a fixed number of neighbors at each layer rather than using the entire neighborhood, making it scalable to large graphs. Second, it concatenates the node's own features with the aggregated neighbor features rather than simply averaging them, allowing the model to distinguish between a node's own identity and its neighborhood context.

```python
from torch_geometric.nn import SAGEConv

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
```

GraphSAGE is particularly important because it introduced the concept of inductive learning on graphs. Unlike GCN, which requires the entire graph to be present during training (transductive learning), GraphSAGE can generate embeddings for previously unseen nodes, making it practical for dynamic graphs where new nodes are constantly being added.

### Graph Attention Networks (GAT)

Graph Attention Networks, introduced by Velickovic et al. in 2018, bring the attention mechanism to graph neural networks. Instead of using fixed weights determined by the graph structure (as in GCN) or simple aggregation functions (as in GraphSAGE), GAT learns to assign different importance weights to different neighbors through a self-attention mechanism.

The attention coefficient between nodes i and j is computed as:

```
alpha_ij = softmax_j(LeakyReLU(a^T [W*h_i || W*h_j]))
```

where W is a learned weight matrix, a is a learned attention vector, and || denotes concatenation. The softmax ensures that the attention weights over all neighbors sum to one.

GAT's advantage is that it can learn which neighbors are most relevant for a given task, potentially capturing more nuanced relationships than fixed aggregation schemes.

### Message Passing Neural Networks (MPNN)

The MPNN framework, introduced by Gilmer et al. in 2017, generalizes the message passing paradigm by allowing arbitrary message functions and update functions. This framework is particularly popular in computational chemistry, where edge features (representing bond types, distances, and angles) carry important information.

```python
from torch_geometric.nn import MessagePassing

class CustomMPNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j], dim=-1))
```

![Graph neural networks processing relational data structures](/images/blogs/pool-ml/3.jpg)

## GNNs in Social Network Analysis

Social networks are among the most natural application domains for GNNs. Platforms like Facebook, Twitter, LinkedIn, and Instagram manage graphs with billions of nodes and edges, and GNNs have become essential tools for a wide range of tasks on these platforms.

### Community Detection

Community detection, the task of identifying densely connected groups of users, is fundamental to social network analysis. Traditional approaches like the Louvain algorithm or spectral clustering operate directly on the graph structure without considering node features. GNNs can incorporate both structural and feature information to produce more meaningful communities.

A GNN-based approach to community detection typically works by learning node embeddings that place nodes in the same community close together in the embedding space. The model can be trained with a combination of objectives: a reconstruction loss that encourages the embeddings to preserve the graph structure, and a clustering loss that encourages the embeddings to form distinct clusters.

In practice, this has been used to identify interest-based communities, detect echo chambers, and understand information flow patterns within social networks. Pinterest, for example, has used GNN-based approaches to organize its content into coherent clusters that improve recommendation quality.

### Link Prediction and Friend Recommendation

Link prediction, predicting which edges are likely to form in the future, is directly applicable to friend recommendation systems. GNNs excel at this task because they can learn complex patterns that go beyond simple heuristics like common neighbors or Jaccard similarity.

A GNN-based link prediction system typically works by computing embeddings for each node using a multi-layer GNN, then scoring potential edges using a scoring function applied to pairs of node embeddings. The scoring function might be as simple as a dot product or as complex as a learned neural network.

LinkedIn has published extensively about their use of GNNs for connection recommendations. Their system processes a graph with hundreds of millions of nodes and billions of edges, using GraphSAGE-style sampling to make the computation tractable. The key insight is that GNNs can capture not just who your friends are, but the structural patterns of your social neighborhood, the friends of your friends, the communities you are embedded in, and how your local network topology compares to others.

### Influence Maximization

Influence maximization is the problem of selecting a small set of seed nodes that will maximize the spread of information through the network. This has direct applications in viral marketing, public health campaigns, and misinformation containment.

Traditional approaches to influence maximization are computationally expensive, often requiring Monte Carlo simulations to estimate the influence spread of each potential seed set. GNNs can learn to approximate the influence function directly, predicting the expected spread of a given seed set without expensive simulations.

Recent work has shown that GNN-based approaches can achieve near-optimal solutions while being orders of magnitude faster than traditional methods. The GNN learns to identify structural features that correlate with high influence, such as nodes that bridge different communities or sit at the center of dense clusters.

### Fraud Detection

Financial and social network fraud detection is another area where GNNs have proven highly effective. Fraudulent accounts and transactions often exhibit distinctive network patterns. Fake accounts tend to form densely connected clusters, coordinate their behavior in ways that create unusual temporal patterns, and interact with legitimate accounts in atypical ways.

GNNs can capture these patterns by learning representations that encode both the features of individual accounts (posting behavior, account age, profile completeness) and their network context (who they follow, who follows them, how their neighborhood is structured).

Alibaba has published work on using GNNs for fraud detection in their e-commerce platform, processing graphs with billions of edges to identify fraudulent merchants and reviews. Their approach combines heterogeneous graph neural networks (which handle multiple types of nodes and edges) with attention mechanisms to focus on the most suspicious patterns.

![Molecular graphs enabling AI-driven drug discovery](/images/blogs/pool-ml/4.jpg)

## GNNs in Drug Discovery

Drug discovery is arguably the domain where GNNs have had the most scientifically significant impact. The process of discovering new drugs involves understanding how small molecules interact with biological targets, and this problem is fundamentally about graphs.

### Molecular Property Prediction

A molecule can be naturally represented as a graph where atoms are nodes and chemical bonds are edges. Node features might include the atom type, charge, and hybridization state, while edge features might encode the bond type (single, double, triple, aromatic) and stereochemistry.

GNNs have become the state-of-the-art approach for predicting molecular properties from structure. Given a molecular graph, a GNN processes the graph through multiple message passing layers, then aggregates the node representations into a single graph-level representation using a readout function (such as sum or attention-weighted pooling). This graph-level representation is then passed through a feedforward network to predict the desired property.

```python
from torch_geometric.nn import GINConv, global_add_pool

class MolecularPropertyPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
        ))
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels)
                )
            ))
        self.predictor = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        x = global_add_pool(x, batch)
        return self.predictor(x)
```

Properties that can be predicted include solubility, toxicity, binding affinity, blood-brain barrier permeability, and many others. The MoleculeNet benchmark suite provides standardized datasets for evaluating molecular property prediction models, and GNN-based methods consistently achieve top performance.

### Virtual Screening

Virtual screening is the process of computationally evaluating large libraries of molecules to identify those most likely to bind to a target protein. Traditional approaches like molecular docking are computationally expensive, often requiring minutes to hours per molecule. GNNs can serve as fast surrogate models, screening millions of molecules in seconds.

The typical workflow involves training a GNN on a set of molecules with known binding affinities to a target protein, then using the trained model to score a much larger library of candidate molecules. The top-scoring candidates are then validated experimentally or with more detailed computational methods.

This approach has been used successfully in real drug discovery programs. For example, researchers have used GNN-based virtual screening to identify novel inhibitors for various disease targets, including kinases involved in cancer, proteases involved in viral replication, and receptors involved in neurological disorders.

### De Novo Molecular Generation

Beyond predicting properties of existing molecules, GNNs can be used to generate entirely new molecules with desired properties. This is typically framed as a graph generation problem, where the model learns to construct molecular graphs atom by atom or fragment by fragment.

Several approaches have been proposed for molecular generation. Variational autoencoders (VAEs) with GNN encoders learn a continuous latent space of molecules, allowing interpolation between known drugs to discover new candidates. Reinforcement learning approaches treat molecule construction as a sequential decision problem, rewarding the agent for generating molecules with high predicted activity and drug-likeness. Flow-based models learn invertible transformations between a simple distribution and the distribution of molecular graphs.

One particularly exciting direction is the use of GNNs for protein-ligand co-design, where the model generates molecules specifically tailored to bind a given protein structure. This requires the GNN to process both the molecular graph and the protein structure (often represented as a 3D point cloud or a residue-level graph), learning representations that capture the complementarity between the ligand and its binding site.

### Protein Structure and Interaction

The success of AlphaFold in predicting protein structures demonstrated the power of geometric deep learning applied to biological macromolecules. While AlphaFold uses a specialized architecture based on attention mechanisms and equivariant transformations, many related problems in structural biology are being tackled with GNN-based approaches.

Protein-protein interaction prediction is one such problem. Given the structures of two proteins, can we predict whether and how they will interact? GNNs that operate on the residue-level contact graphs of proteins have shown promising results, learning to identify interface residues and predict binding affinities.

Enzyme function prediction is another application where GNNs have made significant contributions. By representing enzymes as graphs and learning from the relationship between structure and function, GNNs can predict the catalytic activity of enzymes, aid in the design of new enzymes with desired properties, and help annotate the vast number of protein sequences with unknown function.

## Beyond Social Networks and Drug Discovery

The applications of GNNs extend far beyond these two domains. Here are several other areas where GNNs are making a significant impact.

### Recommendation Systems

E-commerce platforms and content providers use GNNs to model the complex relationships between users, products, and interactions. By representing the user-item interaction history as a bipartite graph, GNNs can learn representations that capture collaborative filtering signals more effectively than traditional matrix factorization approaches.

Pinterest's PinSage system, which uses GraphSAGE-style message passing on a graph with billions of pins and users, was one of the first large-scale industrial deployments of GNNs. It significantly improved the quality of related pin recommendations and has been widely influential in the industry.

### Traffic and Transportation

GNNs are used to model traffic flow in road networks, predict travel times, and optimize routing. The road network is naturally a graph, and GNNs can learn spatiotemporal patterns that capture how congestion propagates through the network. Google Maps uses GNN-based models to predict estimated arrival times, combining the graph structure of the road network with temporal features to account for time-varying traffic patterns.

### Physics Simulation

GNNs have shown remarkable ability to learn physical simulations by modeling interacting particles as nodes in a graph. DeepMind's Graph Network Simulator can learn to simulate complex physical systems including fluids, rigid bodies, and deformable materials by processing the system state as a graph and predicting the next-step dynamics.

This approach is particularly exciting because it can generalize to systems with different numbers of particles, different boundary conditions, and even different materials, all using the same trained model. The key is that the physical interactions are local and can be captured by message passing between nearby particles.

### Combinatorial Optimization

Many important optimization problems, from scheduling to routing to chip design, can be formulated on graphs. GNNs have been used to learn heuristics for these problems, often achieving near-optimal solutions much faster than traditional solvers.

For example, GNNs have been applied to the traveling salesman problem, graph coloring, maximum independent set, and satisfiability problems. While they may not always match the performance of specialized solvers, they offer a compelling trade-off between solution quality and computation time, especially for large instances where exact methods are intractable.

![Real-world applications of graph-based machine learning](/images/blogs/pool-ml/5.jpg)

## Practical Challenges and Best Practices

Working with GNNs in practice involves several challenges that are not always apparent from the research literature.

### Over-Smoothing

As the number of GNN layers increases, node representations tend to converge to similar values, a phenomenon known as over-smoothing. This limits the effective depth of GNNs and the size of the neighborhood that each node can incorporate.

Several strategies have been proposed to mitigate over-smoothing. Residual connections, as in ResNet, allow information to bypass message passing layers. Jumping knowledge networks aggregate representations from all layers rather than just the final layer. DropEdge randomly removes edges during training to prevent information from spreading too uniformly.

### Scalability

Real-world graphs can have billions of nodes and edges, far too large to process in a single forward pass. Mini-batch training on graphs requires special techniques because nodes are interconnected, sampling a batch of nodes requires including their neighbors as well, and those neighbors' neighbors, creating an exponential expansion of the computation graph.

GraphSAGE's neighbor sampling, ClusterGCN's graph partitioning, and GraphSAINT's subgraph sampling are all strategies for making GNN training scalable. In production systems, distributed training across multiple machines is often necessary, with frameworks like DistDGL and PyTorch Geometric's distributed data loading providing the necessary infrastructure.

### Expressiveness

Not all GNNs are equally powerful in terms of what graph patterns they can distinguish. The Weisfeiler-Leman graph isomorphism test provides a theoretical framework for understanding GNN expressiveness. Standard message passing GNNs are at most as powerful as the 1-WL test, meaning there are structurally distinct graphs that they cannot tell apart.

Higher-order GNNs, which operate on tuples of nodes rather than individual nodes, can capture more complex structural patterns but at a higher computational cost. In practice, the expressiveness of standard GNNs is often sufficient, but for tasks that require distinguishing subtle structural differences, more powerful architectures may be needed.

### Handling Heterogeneous Graphs

Many real-world graphs contain multiple types of nodes and edges. A knowledge graph might have person, organization, and location nodes connected by works_at, located_in, and founded edges. Heterogeneous GNNs extend the message passing framework to handle different types of nodes and edges with different learned transformations, allowing the model to capture the semantics of different relationship types.

## Looking Ahead

The field of graph neural networks is evolving rapidly. Several exciting directions are worth watching.

Geometric GNNs that incorporate 3D spatial information are becoming increasingly important for molecular and materials science applications. Equivariant neural networks, which respect the symmetries of physical systems (rotation, translation, reflection), are producing more accurate predictions with less training data.

Foundation models for graphs are an emerging concept, with researchers exploring whether the pre-training and transfer learning paradigm that has been so successful in NLP and computer vision can be applied to graph-structured data. Early results suggest that pre-training GNNs on large unlabeled graphs can improve performance on downstream tasks, but the best pre-training strategies are still being developed.

The integration of GNNs with large language models is another frontier. Knowledge graphs can provide structured factual information to ground LLM outputs, while LLMs can help populate and maintain knowledge graphs. This symbiosis between structured and unstructured knowledge representations may prove to be a powerful paradigm for building more reliable AI systems.

Graph neural networks have established themselves as an essential tool in the machine learning practitioner's toolkit. By operating directly on relational data, they unlock applications that were previously intractable and provide insights that emerge only when we model the connections between entities, not just the entities themselves. Whether you are analyzing social networks, designing new drugs, optimizing supply chains, or simulating physical systems, GNNs offer a powerful and flexible framework for learning from the rich structure of graph-structured data.
