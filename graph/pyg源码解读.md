### 注意：

1. 很多时候计算返回的是edge_index, edge_attr, 所以如果用不到edge_attr的话接收时用_替代
2. 当参数中含有num_nodes，输入与不输入皆标记好，保证节点的个数和当前edge_index的个数对得上
3. 涉及子图时，下面很多计算方法存在问题，谨慎



### 函数：

[`degree`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.degree)

计算edge_index的度。输入的是单向的egde_index，如egde_index[0]或egde_index[1]。注意最好输入参数num_nodes，不然默认是*max_val + 1*可能会出现错误

[`softmax`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.softmax)

计算稀疏矩阵的softmax

[`dropout_adj`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.dropout_adj)

丢弃概率p，注意其中的force_undirected参数，打开时会保证丢弃时双向边会同时丢弃

[`sort_edge_index`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.sort_edge_index)

按edge_index[0]的顺序对edge_index进行排序，注意参数`sort_by_row: bool = True`

注：此处不加num_nodes, 不然会出现排序错误问题

```
# by row
tensor([[   0,    0,    0,  ..., 2707, 2707, 2707],
        [ 633, 1862, 2582,  ...,  598, 1473, 2706]])
# by col     
tensor([[ 633, 1862, 2582,  ...,  598, 1473, 2706],
        [   0,    0,    0,  ..., 2707, 2707, 2707]])
        
# 错误示例
tensor([[   0,    0,  633, 1862,    1],
        [ 633, 1862,    0,    0,    1]])
tensor([[   1,    0,    0,  633, 1862],
        [   1,  633, 1862,    0,    0]])
```

[`coalesce`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.coalesce)

按edge_index[0]的顺序对edge_index进行排序并删除其重复的条目（可以看作是[`sort_edge_index`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.sort_edge_index)的加强版本）。edge_attr中的重复条目通过根据给定的 reduce 选项将它们进行合并

[`is_undirected`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.is_undirected)

判断图是否是无向图

[`to_undirected`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.to_undirected)

把图转为无向图

[`contains_self_loops`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.contains_self_loops)

判断图是否包括自环

[`remove_self_loops`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.remove_self_loops)

移除自环

[`segregate_self_loops`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.segregate_self_loops)

隔离出自环，返回四个参数

```
edge_index, edge_attr, loop_edge_index, loop_edge_attr
```

[`add_self_loops`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.add_self_loops)

增加自环，返回的是乱序的edge_index，可以使用`sort_edge_index`变为有序

（存在一个问题，当是子图进行自环且输入num_nodes参数添加时，出现如下情况）不输入num_nodes时会多出很多不需要的自环节点

```
tensor([[   0,    0,  633, 1862],
        [ 633, 1862,    0,    0]])
tensor([[   0,    0,  633, 1862,    0,    1,    2,    3],
        [ 633, 1862,    0,    0,    0,    1,    2,    3]])
        
修正方法：将
loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
改为
loop_index = torch.unique(torch.cat([edge_index[0], edge_index[1]], dim=0))
`add_remaining_self_loops`同上
```

[`add_remaining_self_loops`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.add_remaining_self_loops)

相对于`add_self_loops`，针对的是当图中存在部分自环时，不会和`add_self_loops`一样进行重复添加，同样存在子图问题

```
tensor([[   0,    0,    1,  633, 1862],
        [ 633, 1862,    1,    0,    0]])
# `add_self_loops`
tensor([[   0,    0,    1,  633, 1862,    0,    1,    2,    3,    4],
        [ 633, 1862,    1,    0,    0,    0,    1,    2,    3,    4]])
# `add_remaining_self_loops`    
tensor([[   0,    0,  633, 1862,    0,    1,    2,    3,    4],
        [ 633, 1862,    0,    0,    0,    1,    2,    3,    4]])
```

[`get_self_loop_attr`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.get_self_loop_attr)

2.0.4版本还没有这个功能

返回自环的edge_attr，对于edge_index不存在的自环返回0，对于没有edge_attr的自环返回1

[`contains_isolated_nodes`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.contains_isolated_nodes)

判断图是否存在孤立点

[`remove_isolated_nodes`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.remove_isolated_nodes)

移除孤立点，自环视为非孤立点，子图时存在问题

```
tensor([[   0,    0,  633, 1862,    1],
        [ 633, 1862,    0,    0,    1]])

(tensor([[0, 0, 1, 2],
        [1, 2, 0, 0]]), None, tensor([ True, False, False,  ..., False, False,  True]))
```

[`get_num_hops`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.get_num_hops)

返回模型从中聚合信息的跳数

[`subgraph`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.subgraph)

输入edge_index，根据Bool/Long矩阵subset来决定留下来的子图的edge_index，源码中有如下语句：

```
node_mask = subset
edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
edge_index = edge_index[:, edge_mask]
```

所以无向图如果存在一条边已经被移除，那么另一条也会被移除，因此如果使用subgraph来生成丢弃节点的话要注意输入subset的问题。

注意参数*relabel_nodes*标签

k_hop_subgraph

返回subset, edge_index, inv, edge_mask

Computes the k-hop subgraph of `edge_index` around node `node_idx`. It returns (1) the nodes involved in the subgraph, (2) the filtered `edge_index` connectivity, (3) the mapping from node indices in `node_idx` to their new location, and (4) the edge mask indicating which edges were preserved.

其中inv是和`node_idx`相同大小的向量，对应的是其在新位置上的节点标签



[`get_laplacian`](https://pytorch-geometric.readthedocs.io/en/2.1.0/modules/utils.html#torch_geometric.utils.get_laplacian)

Computes the graph Laplacian of the graph given by `edge_index` and optional `edge_weight`.

计算图拉普拉斯矩阵，相较于传统的矩阵模式，edge_index代表邻接矩阵，edge_weight代表邻接矩阵的边权重

