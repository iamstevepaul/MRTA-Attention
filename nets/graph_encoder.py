import torch
import numpy as np
from torch import nn
import math

class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):

        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        h = self.layers(h)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )

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
        concat = torch.cat((F0_embedding_3d[:, :, None, :], neighbour_delta_embedding), 2)

        F_embed_final = self.final_embedding(concat).sum(dim=2)
        init_depot_embed = self.init_embed_depot(X['depot'])[:, None, :]
        h = torch.cat((init_depot_embed, F_embed_final), -2)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


class CCN2(nn.Module):

    def __init__(
            self,
            node_dim = 3,
            embed_dim = 128,
            number_of_nodes = 200
    ):
        super(CCN2, self).__init__()
        self.W0 = torch.nn.Linear(node_dim, embed_dim)
        self.W1 = torch.nn.Linear(embed_dim, embed_dim)
        self.W2 = torch.nn.Linear(embed_dim, embed_dim)
        self.neighbour_threshold_distance =  0.055
        self.embed_dim = embed_dim
        self.number_of_nodes = number_of_nodes

    # def forward(self, node_locations, time_deadline, depot):
    def forward(self, input):
        # depot = torch.rand([1, 2])
        node_locations = input['loc']
        time_deadline = input['deadline']
        depot = input['depot']
        # locations = torch.cat((depot, node_locations), dim=0)
        locations = torch.cat((depot[:,None, :], node_locations), dim=1)
        time_deadline = torch.cat((torch.zeros(time_deadline.size()[0], 1, device=time_deadline.device), time_deadline), dim=1)
        # start_time = time.time()

        # define neighbouring point

        distance_matrix = (locations[:, :, None, :] - locations[:, None, :, :]).norm(p=2, dim=-1)
        neighbour_matrix = (distance_matrix <= self.neighbour_threshold_distance).double()
        vertices = torch.from_numpy(np.arange(0, self.number_of_nodes + 1))

        dt = self.W0(torch.cat([locations, time_deadline[:,:, None]], -1))
        rl = torch.nn.ReLU()
        fv_0 = rl(dt)
        Fv_2 = []
        omega_0 = vertices
        for samp in range(locations.size()[0]):


            omega_1 = []
            for i in vertices:
                i_neighbours = torch.nonzero(neighbour_matrix[samp, i], as_tuple=False).view(-1)
                omega_1.append(i_neighbours)
            neighbors = omega_1

            omega_2 = []
            for i in vertices:
                omega_2_i = set([])
                for j in neighbors[i]:
                    omega_2_i = omega_2_i.union(set(omega_1[j].tolist()))
                omega_2.append(omega_2_i)

            fv_1 = []
            for v in vertices:
                omega_1_v = torch.tensor(list(omega_1[v]), device=time_deadline.device)
                phi_1_vw = torch.zeros((omega_1_v.size()[0], self.embed_dim), device=time_deadline.device)
                for w in omega_1_v:
                    omega_0_w = omega_0[w]
                    X_1_vw = torch.zeros([len(omega_1_v), 1], device=time_deadline.device)

                    for i in range(len(omega_1_v)):
                        v_o1 = omega_1_v[i]
                        # for j in range(1):
                        #     w_o0 = omega_0_w[j]
                        if v_o1 == omega_0_w:
                            X_1_vw[i, 0] = 1
                    phi_1_vw += torch.matmul(X_1_vw, fv_0[samp,omega_0_w].unsqueeze(0))
                fv_1.append(phi_1_vw.sum(dim=0))
            fv_1 = torch.stack(fv_1)
            fv_2 = []

            for v in vertices:
                omega_2_v = torch.tensor(list(omega_2[v]), device=time_deadline.device)
                phi_2_vw = torch.zeros((omega_2_v.size()[0], self.embed_dim), device=time_deadline.device)
                for w in omega_2_v:
                    omega_1_w = omega_1[w]
                    X_2_vw = torch.zeros([len(omega_2_v), len(omega_1_w)], device=time_deadline.device)
                    for i in range(len(omega_2_v)):
                        v_o2 = omega_2_v[i]
                        for j in range(len(omega_1_w)):
                            w_o1 = omega_1_w[j]
                            if v_o2 == w_o1:
                                X_2_vw[i, j] = 1

                    phi_2_vw += torch.matmul(X_2_vw, fv_1[omega_1_w])
                fv_2.append(phi_2_vw.sum(dim=0))
            # print(time.time() - start_time, ' Seconds')
            fv_2 = torch.stack(fv_2)
            Fv_2.append(fv_2)

        Fv_2 = torch.stack(Fv_2)
        rl = nn.ReLU()
        Fv_2 = rl(self.W2(Fv_2))
        return (
            Fv_2,  # (batch_size, graph_size, embed_dim)
            Fv_2.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )

class CCN(nn.Module):
    def __init__(
            self,
            node_dim = 3,
            embed_dim = 128,
            n_layers = 2,
    ):
        super(CCN, self).__init__()
        self.init_embed_3d = nn.Linear(node_dim, embed_dim)
        self.init_embed_2d = nn.Linear(2, embed_dim)
        self.neighbor_embed = nn.Linear(int(embed_dim * 6), embed_dim)
        self.init_embed_depot = nn.Linear(2, embed_dim)

    def forward(self, X, mask = None):
        x = torch.cat((X['loc'], X['deadline'][:,:, None]), 2)
        x2 = x[:,:,0:2]
        activ = nn.LeakyReLU()
        F0_embedding_2d = self.init_embed_2d(x2)
        F0_embedding_3d = self.init_embed_3d(x)
        # F0_embedding.reshape([1])


        dist_mat = (x2[:, None] - x2[:,:,None]).norm(dim=-1, p=2) ## device to cuda to be added
        neighbors = dist_mat.sort().indices[:,:,:6] # for 6 neighbours
        NB_embed  = F0_embedding_2d[:, neighbors][0]
        size_nb_embed = NB_embed.size()
        NB_embed_c = F0_embedding_2d[:, neighbors][0].reshape([size_nb_embed[0],size_nb_embed[1],-1])
        F_neigh_embed = self.neighbor_embed(NB_embed_c)
        F_embed_final = F_neigh_embed + F0_embedding_3d
        init_depot_embed = self.init_embed_depot(X['depot'])[:, None, :]
        h = activ(torch.cat((init_depot_embed, F_embed_final), -2))
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )
        # omega1 = neighbors[neighbors].reshape(self.n_nodes, 36)

