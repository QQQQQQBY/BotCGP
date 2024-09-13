import torch

# select the edge type of following and friends in MGTAB
edge_index = torch.load("MGTAB/edge_index.pt").T.tolist()
edge_type = torch.load("MGTAB/edge_type.pt").tolist()

edge_type_01 = []
edge_index_01 = []
for i in range(edge_type.shape[0]):
    if edge_type[i] == 0:
        edge_index_01.append(edge_index[i])
        edge_type_01.append(0)
    if edge_type[i] == 1:
        edge_index_01.append(edge_index[i])
        edge_type_01.append(1)

edge_index_01 = torch.tensor(edge_index_01).T
edge_type_01 = torch.tensor(edge_type_01)

torch.save(edge_type_01,"Dataset/MGTAB/edge_type_01.pt")
torch.save(edge_index_01,"Dataset/MGTAB/edge_index_01.pt")