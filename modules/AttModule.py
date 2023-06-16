import torch
import torch.nn as nn

class AttentionModule(torch.nn.Module):
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]

    def __init__(self,n_features ):
        '''n_features: [0]: 5
                       [1]: 1024
        '''
        super().__init__()
        #TODO
        self.n_feat = n_features

        self.k_layer = nn.Linear(self.n_feat[1],1,bias=False)
        self.q_layer = nn.Linear(self.n_feat[1],1,bias=False)

        self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, x, f):
        #x audio
        #f video
        k = self.k_layer(x)
        q = self.q_layer(x)

        e = torch.matmul(k,q.transpose(dim0=1,dim1=2)) / torch.sqrt(torch.tensor(self.n_feat[1]))
        e_post = torch.triu(e)
        e_post[e_post == 0] = -torch.inf
        a = self.softmax_layer(e_post)

        f_out = torch.matmul(a.transpose(dim0=1,dim1=2),f)
        return f_out
