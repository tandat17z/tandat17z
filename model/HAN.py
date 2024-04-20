import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from model.SentenceAttention import SentenceAttention

class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim, dropout):
        """
        :param num_classes: number of classes
        :param vocab_size: number of words in the vocabulary of the model
        :param embed_dim: dimension of word embeddings
        :param word_gru_hidden_dim: dimension of word-level GRU; biGRU output is double this size
        :param sent_gru_hidden_dim: dimension of sentence-level GRU; biGRU output is double this size
        :param dropout: dropout rate; 0 to not use dropout
        """
        super(HierarchicalAttentionNetwork, self).__init__()

        self.sent_attention = SentenceAttention(
            vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim, dropout)

        # classifier
        self.fc = nn.Linear(2 * sent_gru_hidden_dim, num_classes)
        self.out = nn.LogSoftmax(dim=-1) 

        self.dropout = dropout

    def forward(self, docs, doc_lengths, sent_lengths):
        """
        :param docs: encoded document-level data; LongTensor (num_docs, padded_doc_length, padded_sent_length)
        :param doc_lengths: unpadded document lengths; LongTensor (num_docs)
        :param sent_lengths: unpadded sentence lengths; LongTensor (num_docs, max_sent_len)
        :return: class scores, attention weights of words, attention weights of sentences
        """
        doc_embeds, word_att_weights, sent_att_weights = self.sent_attention(docs, doc_lengths, sent_lengths)
        scores = self.out(self.fc(doc_embeds))

        return scores, word_att_weights, sent_att_weights

