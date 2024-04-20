import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

class WordAttention(nn.Module):
    """
    Word-level attention module.
    """

    def __init__(self, vocab_size, embed_dim, gru_hidden_dim, dropout):
        super(WordAttention, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        # output (batch, hidden_size)
        self.gru = nn.GRU(embed_dim, 
                          gru_hidden_dim, 
                          num_layers=1, 
                          batch_first=True, 
                          bidirectional=True,
                          dropout=dropout)

        self.layer_norm = nn.LayerNorm(2 * gru_hidden_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

        # Maps gru output to `att_dim` sized tensor
        self.attention = nn.Linear(2 * gru_hidden_dim, 2 * gru_hidden_dim)

        # Word context vector (u_w) to take dot-product with
        self.context_vector = nn.Linear(2 * gru_hidden_dim, 1, bias=False)

    def init_embeddings(self, embeddings):
        """
        Initialized embedding layer with pretrained embeddings.
        :param embeddings: embeddings to init with
        """
        # NOTE MODIFICATION (EMBEDDING)
        self.embeddings.weight = nn.Parameter(embeddings)

    def freeze_embeddings(self, freeze=False):
        """
        Set whether to freeze pretrained embeddings.
        :param freeze: True to freeze weights
        """
        # NOTE MODIFICATION (EMBEDDING)
        self.embeddings.weight.requires_grad = not freeze

    def forward(self, sents, sent_lengths):
        """
        :param sents: encoded sentence-level data; LongTensor (num_sents, pad_len)
        :param sent_lengths: sentence lengths; LongTensor (num_sents)
        :return: sentence embeddings, attention weights of words
        """
        # Sort sents by decreasing order in sentence lengths
        sent_lengths, sent_perm_idx = sent_lengths.sort(dim=0, descending=True)
        sents = sents[sent_perm_idx]

        sents = self.embeddings(sents)
        sents = self.dropout(sents)

        packed_words = pack_padded_sequence(sents, lengths=sent_lengths.tolist(), batch_first=True)
        valid_bsz = packed_words.batch_sizes

        # Apply word-level GRU over word embeddings
        packed_words, _ = self.gru(packed_words)
        normed_words = self.layer_norm(packed_words.data)

        # Tính softmax ----------------------------------------------------------------------------
        # normed_words.shape = (num_words, 2 * sent_gru_hidden_dim)
        att = torch.tanh(self.attention(normed_words.data))
        att = self.context_vector(att).squeeze(1)

        # val = att.max()
        # att = torch.exp(att - val) # att.size: (n_words)
        att = torch.exp(att)
        att, _ = pad_packed_sequence(PackedSequence(att, valid_bsz), batch_first=True)# Restore as sentences by repadding
        att_weights = att / torch.sum(att, dim=1, keepdim=True)

        # Restore as sentences by repadding
        sents, _ = pad_packed_sequence(packed_words, batch_first=True)
        sents = sents * att_weights.unsqueeze(2)# Compute sentence vectors
        sents = sents.sum(dim=1)

        # Restore the original order of sentences (undo the first sorting)
        _, sent_unperm_idx = sent_perm_idx.sort(dim=0, descending=False)
        sents = sents[sent_unperm_idx] 
        att_weights = att_weights[sent_unperm_idx]

        return sents, att_weights
