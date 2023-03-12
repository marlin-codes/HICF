import torch
import torch.nn as nn
from torch.nn.modules.module import Module


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, network, num_layers):
        super(HyperbolicGraphConvolution, self).__init__()
        self.agg = HypAgg(manifold, c_in, out_features, network, num_layers)

    def forward(self, input):
        x, adj = input
        h = self.agg.forward(x, adj)
        output = h, adj
        return output


class StackGCNs(Module):

    def __init__(self, num_layers):
        super(StackGCNs, self).__init__()

        self.num_gcn_layers = num_layers - 1

        self.order_attention = nn.Sequential(
            nn.Linear(4, 1),
            nn.ReLU(inplace=True),
            nn.Linear(1, 4),
            nn.Sigmoid()
        ).cuda()

    def plainGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            output.append(torch.spmm(adj, output[i]))
        return output[-1]

    def attentive_output(self, x):
        x_permute = x.permute(0, 2, 1).to(x.device)
        x_att_permute = self.order_attention(x_permute)
        att = x_att_permute.permute(0, 2, 1)
        x = att*x
        return x.sum(dim=1)

    def resAttSumGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            output.append(torch.spmm(adj, output[i]))
        output = torch.cat([res.unsqueeze(1) for res in output], dim=1)
        att_output = self.attentive_output(output)
        return att_output

    def resSumGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            output.append(torch.spmm(adj, output[i]))
        return sum(output[1:])

    def resAddGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        if self.num_gcn_layers == 1:
            return torch.spmm(adj, x_tangent)
        for i in range(self.num_gcn_layers):
            if i == 0:
                output.append(torch.spmm(adj, output[i]))
            else:
                output.append(output[i] + torch.spmm(adj, output[i]))
        return output[-1]

    def denseGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            if i > 0:
                output.append(sum(output[1:i + 1]) + torch.spmm(adj, output[i]))
            else:
                output.append(torch.spmm(adj, output[i]))
        return output[-1]


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, network, num_layers):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.stackGCNs = getattr(StackGCNs(num_layers), network)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)

        output_tan = self.stackGCNs((x_tangent, adj))

        output = self.manifold.proj(self.manifold.expmap0(output_tan, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)
