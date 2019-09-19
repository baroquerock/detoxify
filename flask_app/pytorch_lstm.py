import torch
from torch import nn
from torch.nn import functional as F


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)
        x = x.permute(0, 3, 2, 1)
        x = super(SpatialDropout, self).forward(x)
        x = x.permute(0, 3, 2, 1)
        x = x.squeeze(2)
        return x


class IdentityLSTM(nn.Module):

    def __init__(self, max_features, n_targets=5, embed_size=300, units=16, num_layers=2):

        super(IdentityLSTM, self).__init__()

        self.embedding = nn.Embedding(max_features+1, embed_size)
        self.embedding_dropout = SpatialDropout(0.2)

        self.lstm = nn.LSTM(embed_size, units, bidirectional=True, dropout=0.2,
                            num_layers=num_layers, batch_first=True)

        self.linear = nn.Linear(units*4, units)
        self.linear_out = nn.Linear(units, n_targets)


    def forward(self, x):

        m_embed = self.embedding(x)
        m_embed = self.embedding_dropout(m_embed)

        m_lstm, _ = self.lstm(m_embed)

        avg_pool = torch.mean(m_lstm, 1) # torch.Size([1, units*2])
        max_pool, _ = torch.max(m_lstm, 1) # torch.Size([1, units*2])
        m_conc = torch.cat((max_pool, avg_pool), 1) # torch.Size([1, units*4])
        m_conc_linear  = F.relu(self.linear(m_conc))
        out = self.linear_out(m_conc_linear)
        return out, 0



class ToxicLSTM(nn.Module):

    def __init__(self, max_features, n_targets=6, embed_size=300, units=64, num_layers=2):

        super(ToxicLSTM, self).__init__()

        self.identity_lstm = IdentityLSTM(max_features)

        self.lstm = nn.LSTM(embed_size, units, bidirectional=True, dropout=0.2,
                            num_layers=num_layers, batch_first=True)

        self.linear = nn.Linear(units * 4, units * 2)
        self.linear_out = nn.Linear(units * 2 + 5, 1)
        self.linear_aux_out = nn.Linear(units * 2 + 5, n_targets)


    def forward(self, x):

        identities, _ = self.identity_lstm(x)

        m_embed = self.identity_lstm.embedding(x)
        m_embed = self.identity_lstm.embedding_dropout(m_embed)
        m_lstm, _ = self.lstm(m_embed)

        avg_pool = torch.mean(m_lstm, 1) # torch.Size([1, units*2])
        max_pool, _ = torch.max(m_lstm, 1) # torch.Size([1, units*2])
        m_conc = torch.cat(( max_pool, avg_pool), 1) # torch.Size([1, units*4+5])

        m_linear = torch.cat((identities, self.linear(m_conc)), 1)
        m_linear = torch.relu(m_linear)
        #hidden = m_conc + m_conc_linear

        result = self.linear_out(m_linear)
        aux_result = self.linear_aux_out(m_linear)
        out = torch.cat([result, aux_result], 1)

        return out, identities