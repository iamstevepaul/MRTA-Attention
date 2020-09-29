import torch
import numpy as np
import time
from torch import nn


class CCN3(nn.Module):

    def __init__(
            self,
            node_dim = 3,
            embed_dim = 128,
            n_layers = 2,
    ):
        super(CCN3, self).__init__()
        self.init_embed = nn.Linear(node_dim, embed_dim)
        self.init_neighbour_embed = nn.Linear(node_dim, embed_dim)
        self.neighbour_encode = nn.Linear(node_dim, embed_dim)
        self.init_embed_depot = nn.Linear(2, embed_dim)
        self.final_embedding = nn.Linear(embed_dim, embed_dim)

    def forward(self, X, mask=None):
        x = torch.cat((X['loc'], X['deadline'][:, :, None]), 2)
        x2 = x[:, :, 0:2]
        activ = nn.LeakyReLU()
        # F0_embedding_2d = self.init_embed_2d(x2)
        F0_embedding_3d = self.init_embed(x)
        # F0_embedding.reshape([1])

        dist_mat = (x2[:, None] - x2[:, :, None]).norm(dim=-1, p=2)  ## device to cuda to be added
        neighbors = dist_mat.sort().indices[:, :, :6]  # for 6 neighbours
        neighbour = x[:, neighbors][0]
        neighbour_delta = neighbour - x[:, :, None, :]
        neighbour_delta_embedding = self.neighbour_encode(neighbour_delta)
        concat = torch.cat((F0_embedding_3d[:,:,None,:], neighbour_delta_embedding), 2)

        F_embed_final = self.final_embedding(concat).sum(dim=2)
        init_depot_embed = self.init_embed_depot(X['depot'])[:, None, :]
        h = torch.cat((init_depot_embed, F_embed_final), -2)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )

class CCN(nn.Module):

    def __init__(
            self,
            node_dim = 3,
            embed_dim = 128,
    ):
        super(CCN, self).__init__()
        self.W0 = torch.nn.Linear(node_dim, embed_dim)
        self.W1 = torch.nn.Linear(node_dim, embed_dim)
        self.neighbour_threshold_distance =  0.040
        self.embed_dim = embed_dim

    def forward(self, node_locations, time_deadline):
        depot = torch.rand([1, 2])
        locations = torch.cat((depot, node_locations), dim=0)
        time_deadline = torch.cat((torch.zeros(1, 1), time_deadline), dim=0)
        start_time = time.time()

        # define neighbouring point

        distance_matrix = (locations[:, None, :] - locations).norm(dim=2)
        neighbour_matrix = (distance_matrix <= self.neighbour_threshold_distance).double()
        vertices = torch.from_numpy(np.arange(0, num_locations + 1))



        omega_0 = vertices

        omega_1 = []
        for i in vertices:
            i_neighbours = torch.nonzero(neighbour_matrix[i], as_tuple=False).view(-1)
            omega_1.append(i_neighbours)
        neighbors = omega_1


        omega_2 = []
        for i in vertices:
            omega_2_i = set([])
            for j in neighbors[i]:
                omega_2_i = omega_2_i.union(set(omega_1[j].tolist()))
            omega_2.append(omega_2_i)

        print(time.time() - start_time, " seconds for omega 0,1,2 + distnce matrix")
        start_time = time.time()
        dt = self.W0(torch.cat([locations, time_deadline], -1))
        rl = torch.nn.ReLU()
        fv_0 = rl(dt)
        fv_1 = []
        for v in vertices:
            omega_1_v = torch.tensor(list(omega_1[v]))
            phi_1_vw = torch.zeros((omega_1_v.size()[0], self.embed_dim))
            for w in omega_1_v:
                omega_0_w = omega_0[w]
                X_1_vw = torch.zeros([len(omega_1_v), 1])

                for i in range(len(omega_1_v)):
                    v_o1 = omega_1_v[i]
                    # for j in range(1):
                    #     w_o0 = omega_0_w[j]
                    if v_o1 == omega_0_w:
                        X_1_vw[i, 0] = 1
                phi_1_vw += torch.matmul(X_1_vw, fv_0[omega_0_w].unsqueeze(0))
            fv_1.append(phi_1_vw.sum(dim=0))
        fv_1 = torch.stack(fv_1)
        fv_2 = []

        for v in vertices:
            omega_2_v = torch.tensor(list(omega_2[v]))
            phi_2_vw = torch.zeros((omega_2_v.size()[0], self.embed_dim))
            for w in omega_2_v:
                omega_1_w = omega_1[w]
                X_2_vw = torch.zeros([len(omega_2_v), len(omega_1_w)])
                for i in range(len(omega_2_v)):
                    v_o2 = omega_2_v[i]
                    for j in range(len(omega_1_w)):
                        w_o1 = omega_1_w[j]
                        if v_o2 == w_o1:
                            X_2_vw[i, j] = 1

                phi_2_vw += torch.matmul(X_2_vw, fv_1[omega_1_w])
            fv_2.append(phi_2_vw.sum(dim=0))
        print(time.time() - start_time, ' Seconds')
        fv_2 = torch.stack(fv_2)
        return fv_2


if __name__ == "__main__":
    num_locations = 10
    num_samples = 2
    data = {
            'loc': torch.FloatTensor(num_samples, num_locations, 2).uniform_(0, 1),
            'depot': torch.FloatTensor(num_samples,2).uniform_(0, 1),
            'deadline': torch.FloatTensor(num_samples, num_locations).uniform_(0.1, 1)
        }



    # node_locations = torch.rand([num_locations,2])
    # time_deadline = torch.rand([num_locations,1])
    # fv_2 = get_CCN_encoding(node_locations, time_deadline)
    ccn = CCN3()
    fv_2 = ccn(data)

    # torch.save(
    #     {
    #         'vectors': fv_2
    #     }, "vec.pt"
    # )
    # vec2 = torch.load("vec.pt")
    pass







