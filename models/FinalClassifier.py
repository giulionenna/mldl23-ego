from torch import nn
from torch.autograd import Function
import torch
from modules.TRNmodule import RelationModuleMultiScale
from scipy.stats import entropy 

class Classifier(nn.Module):
    def __init__(self, num_class, n_features,temporal_type,ablation_mask,device):
        super().__init__()
        """
        n_features: [0]: 5
                    [1]: 1024
        tmeporal_type: TRN or pooling
        ablation_mask: Dict("gsd":
                             gtd
                             grd
                             )
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """
        self.num_class = num_class
        self.n_feat = n_features
        self.temporal_type = temporal_type
        self.ablation_mask = ablation_mask
        self.batch_size = 32 #TODO *************
        self.device = device
        #GSF
        
        n_gsf_out = 512
        self.n_gsf_out = n_gsf_out
        
        #Method 1
        self.gsf_vec = nn.ModuleList()
        for i in range(self.n_feat[0]):
            gsf = nn.Sequential()
            gsf.add_module('gsf_fc1', nn.Linear(self.n_feat[1], n_gsf_out))
            gsf.add_module('gsf_relu1', nn.LeakyReLU(0.1,inplace=True))
            #gsf.add_module('gsf_drop1', nn.Dropout())
            #gsf.add_module('gsf_fc2', nn.Linear(n_gsf_out, n_gsf_out))
            #gsf.add_module('gsf_relu2', nn.LeakyReLU(0.1))
           
            self.gsf_vec += [gsf]
        self.gsf_bn_source = nn.ModuleList()
        self.gsf_bn_target = nn.ModuleList()
        for i in range(self.n_feat[0]):
            bn_s = nn.BatchNorm1d(n_gsf_out)
            bn_t = nn.BatchNorm1d(n_gsf_out)
            
            self.gsf_bn_source +=[bn_s]
            self.gsf_bn_target +=[bn_t]


        #Spatial Domain Discriminator
        if(ablation_mask["gsd"]):
            n_gsd = 256;
            self.gsd = nn.Sequential()
            self.gsd.add_module('gsd_fc1', nn.Linear(n_gsf_out*n_features[0], n_gsd))
            #self.gsd.add_module('gsd_bn1', nn.BatchNorm1d(n_gsd))
            self.gsd.add_module('gsd_relu1', nn.LeakyReLU(0.1,inplace=True))
            #self.gsd.add_module('gsd_drop1', nn.Dropout())
            #self.gsd.add_module('gsd_fc2', nn.Linear(n_gsd, n_gsd//2))
            #self.gsd.add_module('gsd_bn2', nn.BatchNorm1d( n_gsd//2))
            #self.gsd.add_module('gsd_relu2', nn.LeakyReLU(0.1))      
            
            self.gsd_bn_source = nn.BatchNorm1d( n_gsd)
            self.gsd_bn_target = nn.BatchNorm1d( n_gsd)
            self.gsd_final_layer =  nn.Linear( n_gsd, 2)
            
            
        
        #Temporal Pooling
        if(temporal_type == "TRN"):
            self.trn = nn.Sequential()
            self.trn.add_module('trn', RelationModuleMultiScale(img_feature_dim=n_gsf_out, num_bottleneck=n_gsf_out, num_frames=n_features[0]))
            n_grd_out = 512
            self.n_grd_out = n_grd_out
            if(ablation_mask["grd"]):
                self.grd_all = nn.ModuleList()
                for i in range(self.n_feat[0]-1):
                    grd = nn.Sequential(
                        nn.Linear(n_gsf_out,n_grd_out),
                        #nn.BatchNorm1d(n_grd_out),
                        nn.LeakyReLU(0.1,inplace=True),
                        #nn.Dropout()
                        #nn.Linear(n_grd_out, n_grd_out//2),
                        #nn.BatchNorm1d(n_grd_out//2),
                        #nn.ReLU(True) ,
                        #nn.Linear(n_grd_out//2, 2),
                        
                        )
                    self.grd_all += [grd]
                self.grd_bn_source = nn.ModuleList()
                self.grd_bn_target = nn.ModuleList()
                self.grd_final_layer = nn.ModuleList()
                for i in range(self.n_feat[0]-1):
                    bn_s = nn.BatchNorm1d(n_grd_out)
                    bn_t = nn.BatchNorm1d(n_grd_out)
                    fc_grd = nn.Linear(n_grd_out, 2)
                    self.grd_bn_source +=[bn_s]
                    self.grd_bn_target +=[bn_t]
                    self.grd_final_layer += [fc_grd]


        self.AvgPool = nn.AvgPool2d([5,1])       
        #Temporal Domain discriminator
        if(ablation_mask["gtd"]):
            n_gtd = 512
            #Method 1
            self.gtd = nn.Sequential()
            self.gtd.add_module('gtd_fc1',     nn.Linear(n_gsf_out, n_gtd))
            #self.gtd.add_module('gtd_bn1',     nn.BatchNorm1d(n_gtd))
            self.gtd.add_module('gtd_relu1',   nn.LeakyReLU(0.1,inplace=True))
            #self.gtd.add_module('gtd_drop1',   nn.Dropout())
            #self.gtd.add_module('gtd_fc2',     nn.Linear(n_gtd, n_gtd//2))
            #self.gtd.add_module('gtd_relu2',   nn.LeakyReLU(0.1))      

            self.gtd_bn_source = nn.BatchNorm1d(n_gtd)
            self.gtd_bn_target = nn.BatchNorm1d(n_gtd)

            self.gtd_final_layer = nn.Linear(n_gtd,2)
        #Gy
        #1
        self.gy = nn.Sequential()
       #2
        self.gy.add_module('gy_fc1', nn.Linear(n_gsf_out, num_class))


    def forward(self, x,alpha = 1,domain="source"):
        
        spatial_domain_out = None
        temporal_domain_out = None
        grd_outs = None
        class_out = None
        #GSF 
        # method 1
        gsf_out_pre = torch.zeros(x.shape[0], self.n_feat[0], self.n_gsf_out).to(self.device)
        gsf_out = torch.zeros(x.shape[0], self.n_feat[0], self.n_gsf_out).to(self.device)
        for i in range(self.n_feat[0]):
            gsf_out_pre[:,i,:] = self.gsf_vec[i](x[:,i,:])+0
            if(domain == "source"):
                gsf_out[:,i,:] = self.gsf_bn_source[i](gsf_out_pre[:,i,:]+0)+0
            elif(domain == "target"):
                gsf_out[:,i,:] = self.gsf_bn_target[i](gsf_out_pre[:,i,:]+0)+0
    
        #spatial domain out
        if(self.ablation_mask["gsd"]):
            reverse_features = ReverseLayerF.apply(gsf_out,alpha)
            spatial_domain_out_pre = self.gsd(reverse_features.view(-1,5*self.n_gsf_out))
            if(domain=="source"):
                spatial_domain_out_bn = self.gsd_bn_source(spatial_domain_out_pre)
            elif(domain=="target"):
                spatial_domain_out_bn = self.gsd_bn_target(spatial_domain_out_pre)
            spatial_domain_out = self.gsd_final_layer(spatial_domain_out_bn)
        #temporal aggregation 
        if(self.temporal_type == "TRN"):
            TRN_out = self.trn(gsf_out)
            if(self.ablation_mask["grd"]):
                grd_outs_pre = torch.zeros([gsf_out.shape[0],self.n_feat[0]-1,self.n_grd_out]).to(self.device)
                grd_outs_bn = torch.zeros([gsf_out.shape[0],self.n_feat[0]-1,self.n_grd_out]).to(self.device)
                grd_outs = torch.zeros([gsf_out.shape[0],self.n_feat[0]-1,2]).to(self.device)

                for i in range(0,self.n_feat[0]-1):
                    grd_outs_pre[:,i,:] = self.grd_all[i](ReverseLayerF.apply(TRN_out[:,i,:],alpha))+0
                    if(domain=="source"):
                        grd_outs_bn[:,i,:] = self.grd_bn_source[i](grd_outs_pre[:,i,:]+0)+0
                    elif(domain=="target"):
                        grd_outs_bn[:,i,:] = self.grd_bn_target[i](grd_outs_pre[:,i,:]+0)+0
                    grd_outs[:,i,:] = self.grd_final_layer[i](grd_outs_bn[:,i,:]+0)+0

                if(self.ablation_mask["domainA"]):
                #Calcolo Entropia e Attention Weights

                    softmax = nn.Softmax(dim=2)
                    logsoftmax = nn.LogSoftmax(dim=2)
                    #entropy = torch.sum(-(grd_outs) * torch.log(grd_outs), 2).nan_to_num()
                    entropy = torch.sum(-softmax(grd_outs) * logsoftmax(grd_outs), 2)
                    weights = 1-entropy
                    weights = weights.unsqueeze(2).repeat(1,1,TRN_out.shape[2]) #[32,4] -> [32,4,#feat] on dim=2 repeate the element of the second dim

                    TRN_weighted = (weights+1)*TRN_out
                    temporal_aggregation = nn.AvgPool2d([4,1])(TRN_weighted.unsqueeze(1)).squeeze()
                    #weighted_to_avg = ((weights+1)*TRN_out).transpose(1,2)
                    #temporal_aggregation = nn.AvgPool1d(4)(weighted_to_avg)
                    #temporal_aggregation = temporal_aggregation.squeeze(2)

                else:
                    temporal_aggregation = (nn.AvgPool2d([4,1])(TRN_out.unsqueeze(1))).squeeze()
            else:
                temporal_aggregation = (nn.AvgPool2d([4,1])(TRN_out.unsqueeze(1))).squeeze()
        else:
            temporal_aggregation =self.AvgPool(gsf_out.unsqueeze(1)).squeeze()
        #temporal domain
        if(self.ablation_mask["gtd"]):
            temporal_domain_out_pre =  self.gtd(ReverseLayerF.apply(temporal_aggregation,alpha))
            if(domain=="source"):
                temporal_domain_out_bn = self.gtd_bn_source(temporal_domain_out_pre)
            elif(domain=="target"):
                temporal_domain_out_bn = self.gtd_bn_target(temporal_domain_out_pre)
            temporal_domain_out = self.gtd_final_layer(temporal_domain_out_bn)
        
        class_out = self.gy(temporal_aggregation)

        
        return spatial_domain_out,temporal_domain_out, class_out,grd_outs
      
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
