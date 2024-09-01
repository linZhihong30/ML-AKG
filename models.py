import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import SAGEConv
import clip


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.bn = torch.nn.BatchNorm1d(out_features)
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        # support = torch.matmul(inputs, self.weight)
        # output = torch.matmul(adj, support)

        support = torch.matmul(adj, inputs)
        output = torch.matmul(support, self.weight)

        output = self.dropout(output)
        output = self.bn(output)

        # Apply residual connection with dimension matching
        if inputs.size(-1) != output.size(-1):
            # inputs = torch.matmul(inputs, torch.eye(self.in_features, self.out_features).to(inputs.device))
            inputs = torch.cat((inputs, inputs), dim=1)
            if inputs.size(-1) != self.out_features:
                inputs = torch.matmul(inputs, torch.eye(inputs.size(-1), self.out_features).to(inputs.device))

        if self.bias is not None:
            return output + inputs + self.bias
        else:
            return output + inputs

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=512, t=0, adj_file=None):  # t=0.4 adj_file='data/voc/voc_adj.pkl'
        super(GCNResnet, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048)
        )
        # self.mlp = nn.Linear(1024, 2048)

        self.gc1 = GraphConvolution(in_channel, 1024)  # 300->512
        self.gc2 = GraphConvolution(1024, 2048)  # 512->1024

        self.relu = nn.LeakyReLU(0.2)

        _adj, _ = gen_A(num_classes, t, adj_file)  # size=[20, 20]
        _adj_AK = gen_AK(num_classes, 0.5)  # size=[20, 20]

        self.A = Parameter(torch.from_numpy(_adj).float())
        self.K = Parameter(torch.from_numpy(_adj_AK).float())

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        with torch.no_grad():
            feature = self.model.encode_image(feature)

        # normalized features
        feature = feature / feature.norm(dim=-1, keepdim=True)
        feature = feature.float()
        feature = self.mlp(feature)  # torch.Size([batch_size, 2048])

        inp = inp[0]  # [20, 512]

        adj = gen_adj(self.A).detach()
        adj_AK = gen_adj(self.K).detach()

        adj = 0.6 * adj + 0.4 * adj_AK

        x_gcn = self.gc1(inp, adj)
        x_gcn = self.relu(x_gcn)
        x_gcn = self.gc2(x_gcn, adj)

        x = x_gcn.transpose(0, 1)  # torch.Size([2048, 20])
        x = torch.matmul(feature, x)  # torch.Size([16, 20])
        return x

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.mlp.parameters(), 'lr': lr * lrp},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr}
        ]


def gcn_clip(num_classes, t, adj_file=None, in_channel=512):  # t = 0.4  adj_file='data/voc/voc_adj.pkl'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)  # ViT-B/32  ViT-L/14@336px  RN50x64
    return GCNResnet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel)
