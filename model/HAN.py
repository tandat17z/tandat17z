import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from SentenceAttention import SentenceAttention

class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim,
                word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout):
        """
        :param num_classes: number of classes
        :param vocab_size: number of words in the vocabulary of the model
        :param embed_dim: dimension of word embeddings
        :param word_gru_hidden_dim: dimension of word-level GRU; biGRU output is double this size
        :param sent_gru_hidden_dim: dimension of sentence-level GRU; biGRU output is double this size
        :param word_gru_num_layers: number of layers in word-level GRU
        :param sent_gru_num_layers: number of layers in sentence-level GRU
        :param word_att_dim: dimension of word-level attention layer
        :param sent_att_dim: dimension of sentence-level attention layer
        :param use_layer_norm: whether to use layer normalization
        :param dropout: dropout rate; 0 to not use dropout
        """
        super(HierarchicalAttentionNetwork, self).__init__()

        self.sent_attention = SentenceAttention(
            vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim,
            word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout)

        # classifier
        self.fc = nn.Linear(2 * sent_gru_hidden_dim, num_classes)

        # NOTE MODIFICATION (BUG)
        # self.out = nn.LogSoftmax(dim=-1) # option 1
        # erase this line # option 2

        # # NOTE MODIFICATION (FEATURES)
        self.use_layer_nome = use_layer_norm
        self.dropout = dropout

    def forward(self, docs, doc_lengths, sent_lengths):
        """
        :param docs: encoded document-level data; LongTensor (num_docs, padded_doc_length, padded_sent_length)
        :param doc_lengths: unpadded document lengths; LongTensor (num_docs)
        :param sent_lengths: unpadded sentence lengths; LongTensor (num_docs, max_sent_len)
        :return: class scores, attention weights of words, attention weights of sentences
        """
        doc_embeds, word_att_weights, sent_att_weights = self.sent_attention(docs, doc_lengths, sent_lengths)

        # NOTE MODIFICATION (BUG)
        # scores = self.out(self.fc(doc_embeds)) # option 1
        scores = self.fc(doc_embeds) # option 2

        return scores, word_att_weights, sent_att_weights

