import torch
from torch import nn
from torch.nn import functional as F

LSTM_UNITS = 64
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class ToxicLSTM(nn.Module):

    def __init__(self, n_targets, max_features, embed_size):

        super(ToxicLSTM, self).__init__()

        self.embedding = nn.Embedding(max_features+1, embed_size)
        self.embedding_dropout = SpatialDropout(0.3)

        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)
        self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, n_targets)


    def forward(self, x):

        m_embed = self.embedding(x)
        m_embed = self.embedding_dropout(m_embed)

        m_lstm, _ = self.lstm1(m_embed)
        m_lstm, _ = self.lstm2(m_lstm)

        avg_pool = torch.mean(m_lstm, 1)
        max_pool, _ = torch.max(m_lstm, 1)
        m_conc = torch.cat((max_pool, avg_pool), 1)

        m_conc_linear  = F.relu(self.linear(m_conc))

        hidden = m_conc + m_conc_linear

        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)

        return out