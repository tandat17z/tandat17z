import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from model.WordAttention import WordAttention

class SentenceAttention(nn.Module):
    """
    Sentence-level attention module. Contains a word-level attention module.
    """
    def __init__(self, vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim, dropout):
        super(SentenceAttention, self).__init__()

        # Word-level attention module
        self.word_attention = WordAttention(vocab_size, embed_dim, word_gru_hidden_dim, dropout)

        # Bidirectional sentence-level GRU
        self.gru = nn.GRU(2 * word_gru_hidden_dim, 
                          sent_gru_hidden_dim, 
                          num_layers=1,
                          batch_first=True, 
                          bidirectional=True, 
                          dropout=dropout)

        # NOTE MODIFICATION (FEATURES)
        self.layer_norm = nn.LayerNorm(2 * sent_gru_hidden_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)


        # Sentence-level attention
        self.sent_attention = nn.Linear(2 * sent_gru_hidden_dim, 2 * sent_gru_hidden_dim)

        # Sentence context vector u_s ---> u_i * u_s (không cần bias)
        self.sentence_context_vector = nn.Linear(2 * sent_gru_hidden_dim, 1, bias=False)

    def forward(self, docs, doc_lengths, sent_lengths):
        """
        :param docs: encoded document-level data; LongTensor (num_docs, padded_doc_length, padded_sent_length)
        :param doc_lengths: unpadded document lengths; LongTensor (num_docs)
        :param sent_lengths: unpadded sentence lengths; LongTensor (num_docs, padded_doc_length)
        :return: document embeddings, attention weights of words, attention weights of sentences
        """
        # Sort documents by decreasing order in length -->> để tối ưu hóa tính toán 
        # -->> sắp xếp theo trục 0 nên không ảnh hướng tới thứ tự của từ trong 1 câu
        doc_lengths, doc_perm_idx = doc_lengths.sort(dim=0, descending=True)
        docs = docs[doc_perm_idx]
        sent_lengths = sent_lengths[doc_perm_idx]

        # Xóa bỏ phần pad-sentences và gộp tất cả các câu của docs -----------------------------------
        # --> packed_sents = (data: (nums_sents, padded_sent_length), batch_sizes: 1D) - Tổng số câu trong n docs : num_sents
        packed_sents = pack_padded_sequence(docs, lengths=doc_lengths.tolist(), batch_first=True)
        valid_bsz = packed_sents.batch_sizes 

        # Make a long batch of sentence lengths by removing pad-sentences
        packed_sent_lengths = pack_padded_sequence(sent_lengths, lengths=doc_lengths.tolist(), batch_first=True)


        # Word attention module-----------------------------------------------------------------------
        sents, word_att_weights = self.word_attention(packed_sents.data, packed_sent_lengths.data)
        sents = self.dropout(sents)


        # Sentence-level GRU over sentence embeddings -------------------------------------------------
        # --> normed_sents = PackedSequence( data: (số nums_sents, 2*hidden_gru), batchsize: ...)
        packed_sents, _ = self.gru(PackedSequence(sents, valid_bsz))
        normed_sents = self.layer_norm(packed_sents.data)
        

        # Sentence attention --------------------------------------------------------------------------
        att = torch.tanh(self.sent_attention(normed_sents)) # (num_sents, 2 * sent_gru_hidden_dim)
        att = self.sentence_context_vector(att).squeeze(1) # (num_sents, 1) -> (num_sents)

        # Tính soft max -------------------------------------------------------------------------------
        # val = att.max()
        # att = torch.exp(att - val)
        att = torch.exp(att) #

        # Restore as documents by repadding - Để tính softmax cho từng văn bản với độ dài tương ứng, nếu có pad thì att_exp = 0
        # --> sent_att_weights = (num_docs, padded_doc_length)
        att, _ = pad_packed_sequence(PackedSequence(att, valid_bsz), batch_first=True)
        sent_att_weights = att / torch.sum(att, dim=1, keepdim=True)


        # Restore as documents by repadding -----------------------------------------------
        docs, _ = pad_packed_sequence(packed_sents, batch_first=True)

        # Compute document vectors
        docs = docs * sent_att_weights.unsqueeze(2)
        docs = docs.sum(dim=1)

        # Khôi phục lại thứ tự docs ----------------------------------------------------------
        word_att_weights, _ = pad_packed_sequence(PackedSequence(word_att_weights, valid_bsz), batch_first=True)# Restore as documents by repadding
        _, doc_unperm_idx = doc_perm_idx.sort(dim=0, descending=False) # Restore the original order of documents (undo the first sorting)
        docs = docs[doc_unperm_idx]

        word_att_weights = word_att_weights[doc_unperm_idx]
        sent_att_weights = sent_att_weights[doc_unperm_idx]

        return docs, word_att_weights, sent_att_weights

