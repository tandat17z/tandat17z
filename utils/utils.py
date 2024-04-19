import os
import torch
from tqdm import tqdm
from dataset import collate_fn
from pylab import *
from nltk.tokenize import word_tokenize, sent_tokenize


# NOTE MODIFICATION (REFACTOR)
class MetricTracker(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, summed_val, n=1):
        self.val = summed_val / n
        self.sum += summed_val
        self.count += n
        self.avg = self.sum / self.count


# NOTE MODIFICATION (EMBEDDING)
def get_pretrained_weights(glove_path, corpus_vocab, embed_dim, device):
    """
    Returns 50002 words' pretrained weights in tensor
    :param glove_path: path of the glove txt file
    :param corpus_vocab: vocabulary from dataset
    :return: tensor (len(vocab), embed_dim)
    """
    save_dir = os.path.join(glove_path, 'glove_pretrained.pt')
    if os.path.exists(save_dir):
        return torch.load(save_dir, map_location=device)

    corpus_set = set(corpus_vocab)
    pretrained_vocab = set()
    glove_pretrained = torch.zeros(len(corpus_vocab), embed_dim)
    with open(os.path.join(glove_path, 'glove.6B.100d.txt'), 'rb') as f:
        for l in tqdm(f):
            line = l.decode().split()
            if line[0] in corpus_set:
                pretrained_vocab.add(line[0])
                glove_pretrained[corpus_vocab.index(line[0])] = torch.from_numpy(np.array(line[1:]).astype(np.float))

        # handling 'out of vocabulary' words
        var = float(torch.var(glove_pretrained))
        for oov in corpus_set.difference(pretrained_vocab):
            glove_pretrained[corpus_vocab.index(oov)] = torch.empty(100).float().uniform_(-var, var)
        print("weight size:", glove_pretrained.size())
        torch.save(glove_pretrained, save_dir)
    return glove_pretrained
