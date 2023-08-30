import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, dropout):
        super().__init__()
        assert isinstance(hidden_dims, list)
        if len(hidden_dims) == 0:
            self.fcs = nn.ModuleList([nn.Linear(in_dim, out_dim)])
        else:
            self.num_hidden_layers = len(hidden_dims) - 1
            self.fcs = nn.ModuleList([nn.Linear(in_dim, hidden_dims[0])])
            for i in range(self.num_hidden_layers):
                self.fcs.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.fcs.append(nn.Linear(hidden_dims[-1], out_dim))
        self.act = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for layer in self.fcs[:-1]:
            x = self.act(layer(x))
            x = self.dropout(x)
        x = self.fcs[-1](x)  # prediction layer
        return x


class GraphAttentionLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, alpha=0.2, bias=True):
        super().__init__()

        assert (
            emb_dim % num_heads == 0
        ), "For GAT layer, emb_dim must be dividable by num_heads."
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.alpha = alpha
        self.bias = bias
        self.leakyrelu = nn.LeakyReLU(negative_slope=alpha)
        self.act = nn.ReLU()
        # self.act = nn.ELU()
        self.W = nn.Linear(
            emb_dim, emb_dim, bias=self.bias
        )  # each W_k = [emb_dim, emb_dim/num_heads]
        self.a = nn.Linear(emb_dim, 2 * num_heads, bias=self.bias)

    def forward(self, x, A):
        Wh = self.W(x)  # [B,N,F]
        e = self.get_attention_coeff(Wh)  # [B,H,N,N]
        Wh = self.transform(Wh)  # [B,H,N,F/H]

        inf = torch.tensor(float("-inf")).to(x.device)
        A = A.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        att_coeff = torch.where(A == 1.0, self.leakyrelu(e), inf)
        attention = torch.softmax(att_coeff, dim=-1)  # [B,H,N,N]
        attention = self.nan_to_num(attention)
        retval = torch.matmul(attention, Wh)  # [B,H,N,F/H]
        retval = self.act(self.restore(retval))
        return retval

    def transform(self, tensor):
        # input: [B,N,F] -> output: [B,H,N,F/H]
        new_shape = tensor.size()[:-1] + (self.num_heads, -1)
        tensor = tensor.view(new_shape)  # [B,N,H,F/H]
        return tensor.transpose(1, 2)  # [B,H,N,F/H]

    def restore(self, tensor):
        # input: [B,H,N,F/H] -> output: [B,N,F]
        tensor = tensor.transpose(1, 2)  # [B,N,H,F/H]
        new_shape = tensor.size()[:-2] + (self.emb_dim,)
        return tensor.reshape(new_shape)  # [B,N,F]

    def nan_to_num(self, attention):
        attention = torch.where(
            attention.isnan(), torch.Tensor([0.0]).to(attention.data.device), attention
        )
        return attention

    def get_attention_coeff(self, Wh):
        # input: [B,N,F], output: [B,H,N,N]
        e12 = self.a(Wh)  # [B,N,F] -> [B,N,2*H]
        e12 = self.transform(e12)  # [B,N,2*H] -> [B,H,N,2]
        e1, e2 = e12[:, :, :, 0].unsqueeze(-1), e12[:, :, :, 1].unsqueeze(-1)
        return e1 + e2.transpose(-2, -1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            + f"  activation: {self.act}\n"
            + f"  W: Linear({self.emb_dim} -> {self.emb_dim}, bias={self.bias})\n"
            + f"  a: Linear({self.emb_dim} -> {2*self.num_heads}, bias={self.bias})\n)"
        )
