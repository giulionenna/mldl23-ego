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

class DeepAttentionModule(torch.nn.Module):
    def __init__(self,n_features ,depth):
        '''n_features: [0]: 5
                       [1]: 1024
        '''
        super().__init__()
        #TODO
        self.n_feat = n_features
        self.depth = depth
        self.attentions = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.layerNorm = nn.ModuleList()
        for i in range(self.depth):
            a = AttentionModule(n_features=n_features)
            self.attentions+=[a]

            fc =  nn.Linear(1024,1024)
            self.fcs +=[fc]

            lm = nn.LayerNorm(n_features)
            self.layerNorm +=[lm]

    def forward(self, x, f):
        #x audio
        #f video
        for i in range(self.depth-1):
            att = self.attentions[i](x,x)
            x = x+att
            x = self.layerNorm[i](x)
            x = self.fcs[i](x)
        
        att = self.attentions[self.depth-1](x,f)
        x = x+att
        x = self.layerNorm[self.depth-1](x)
        x = self.fcs[self.depth-1](x)
        
        return x
class MultiHeadAttentionModule(torch.nn.Module):
    def __init__(self,n_features ,depth,n_heads,device):
        '''n_features: [0]: 5
                       [1]: 1024
        '''
        super().__init__()
        #TODO
        self.n_feat = n_features
        self.depth = depth
        self.n_heads = n_heads
        self.device = device

        self.heads = nn.ModuleList()
        for i in range(n_heads):
            h = DeepAttentionModule(n_features,depth=depth)
            self.heads +=[h]
        
    def forward(self, x, f):
        #x audio
        #f video
        f_out = torch.zeros(x.shape).to(device=self.device)
        for i in range(self.n_heads):
            f_out += self.heads[i](x,f)

        return f_out/self.n_heads  
#class AttentionModuleBase(torch.nn.Module):
#    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]#

#    def __init__(self,n_features ):
#        '''n_features: [0]: 5
#                       [1]: 1024
#        '''
#        super().__init__()
#        #TODO
#        self.n_feat = n_features#

#        #

#    def forward(self, x, f):
#        #x audio
#        #f video
#        
#        return f_out
