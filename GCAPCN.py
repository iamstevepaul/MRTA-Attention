import torchimport numpy as npimport timefrom torch import nnclass GCAPCN(nn.Module):    def __init__(self,                 n_layers = 3,                 n_hops = 5,                 n_dim = 128,                 n_p = 3,                 node_dim = 3                 ):        super(GCAPCN, self).__init__()        self.n_layers = n_layers        self.n_hops = n_hops        self.n_dim = n_dim        self.n_p = n_p        self.node_dim = node_dim        self.init_embed = nn.Linear(node_dim, n_dim)        self.W1 = nn.Linear(n_dim*n_p, n_dim)        self.W2 = nn.Linear(n_dim * n_p, n_dim)        self.activ = nn.LeakyReLU()    def forward(self, X, mask=None):        # X = torch.cat((data['loc'], data['deadline']), -1)        X_loc = X[:, :, 0:2]        distance_matrix = (((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5        A = (distance_matrix < .3).to(torch.int8)        num_samples, num_locations, _ = X.size()        D = torch.mul(torch.eye(num_locations).expand((num_samples, num_locations, num_locations)),                      (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))        L = D - A        F0 = self.init_embed(X)        g1 = torch.cat((F0[:,:,:,None], torch.matmul(L,F0)[:,:,:,None], torch.matmul(torch.matmul(L,L), F0)[:,:,:,None]), -1).reshape((num_samples, num_locations, -1))        F1 =   self.activ(self.W1(g1))        g2 = torch.cat((F1[:,:,:,None], torch.matmul(L,F1)[:,:,:,None], torch.matmul(torch.matmul(L,L), F1)[:,:,:,None]), -1).reshape((num_samples, num_locations, -1))        return self.activ(self.W2(g2))if __name__ == '__main__':    num_locations = 100        num_samples = 2    data = {        'loc': torch.FloatTensor(num_samples, num_locations, 2).uniform_(0, 1),        'depot': torch.FloatTensor(num_samples ,2).uniform_(0, 1),        'deadline': torch.FloatTensor(num_samples, num_locations, 1).uniform_(0.1, 1)    }    X = torch.cat((data['loc'], data['deadline']), -1)    # X_loc = X[:,:,0:2]    # distance_matrix = (((X_loc[:, :,None] - X_loc[:, None])**2).sum(-1))**.5    # A = (distance_matrix < .3).to(torch.int8)    #    # D = torch.mul(torch.eye(num_locations).expand((num_samples, num_locations, num_locations)), (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))    # L = D - A    # N_i = A.sum(-1)    # A.sum(-1) - 1    #    # torch.eye(num_locations).expand((num_samples, num_locations, num_locations))    #    # (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations))    enc = GCAPCN()    embed = enc(X)