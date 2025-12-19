import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from matplotlib.colors import ListedColormap
import os


def process_and_save_graph(x, adj, Y, save_dir="./pic"):

    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()


    G = nx.Graph()
    N = x.shape[0]
    G.add_nodes_from(range(N))

    target_node = 1701
    if target_node not in G.nodes:
        print(f"错误：1701号节点不存在（图中仅包含{0}~{N - 1}号节点）")
        return


    edges = list(zip(adj[0].tolist(), adj[1].tolist()))
    G.add_edges_from(edges)



    hop3_neighbors = nx.single_source_shortest_path_length(G, source=target_node, cutoff=2)
    relevant_nodes = list(hop3_neighbors.keys())
    subG = G.subgraph(relevant_nodes)
    print(f"子图统计：{len(relevant_nodes)}个节点，{len(subG.edges())}条边")


    edge_list = list(subG.edges())

    df_edges = pd.DataFrame(edge_list, columns=["source_node", "target_node"])


    sorted_nodes = sorted(relevant_nodes)
    node_label_list = [(node, Y[node]) for node in sorted_nodes]
    df_labels = pd.DataFrame(node_label_list, columns=["node_id", "label"])



    os.makedirs(save_dir, exist_ok=True)


    edges_path = os.path.join(save_dir, "edges.csv")
    labels_path = os.path.join(save_dir, "node_labels.csv")
    df_edges.to_csv(edges_path, index=False)
    df_labels.to_csv(labels_path, index=False)
    print(f"CSV文件已保存：\n- 边列表: {edges_path}\n- 节点标签: {labels_path}")



    sub_labels = [Y[node] for node in subG.nodes()]
    unique_labels = sorted(list(set(sub_labels)))
    cmap = ListedColormap(plt.cm.tab10(np.linspace(0, 1, len(unique_labels))))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    node_colors = [cmap(label_to_idx[Y[node]]) for node in subG.nodes()]


    node_sizes = [800 if node == target_node else 400 for node in subG.nodes()]


    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(subG, k=2.5, seed=42)


    nx.draw_networkx_nodes(
        subG, pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="black",
        alpha=0.8
    )
    nx.draw_networkx_edges(
        subG, pos,
        edge_color="gray",
        width=0.8,
        alpha=0.6
    )


    legend_elements = [
        plt.Line2D(
            [0], [0], marker='o', color='w', markerfacecolor=cmap(i),
            markersize=10, label=f'Label {label}'
        ) for i, label in enumerate(unique_labels)
    ]
    legend_elements.append(
        plt.Line2D(
            [0], [0], marker='o', color='w', markerfacecolor='red',
            markersize=12, markeredgewidth=2, label='Node 1701 (Target)'
        )
    )
    plt.legend(handles=legend_elements, loc="upper right", fontsize=10)


    plt.show()


from dataset import *
dataset = load_synthetic_dataset("./data/", "Cora", train_num=3, combine=True)
x=dataset.x
adj=dataset.edge_index
Y=dataset.y
pic=process_and_save_graph(x, adj, Y)