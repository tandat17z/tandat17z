from dataset import News20Dataset, collate_fn
from utils.utils import *
import os, sys
import webbrowser


class Tester:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.device = next(self.model.parameters()).device

        self.dataset = News20Dataset(config.cache_data_dir, config.vocab_path, is_train=False)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=config.batch_size, shuffle=False,
                                                      collate_fn=collate_fn)

        self.accs = MetricTracker()
        self.best_acc = 0

    def eval(self):
        print("Tester.eval...")
        self.model.eval()
        with torch.no_grad():
            self.accs.reset()

            for (docs, labels, doc_lengths, sent_lengths) in self.dataloader:
                batch_size = labels.size(0)

                docs = docs.to(self.device)
                labels = labels.to(self.device)
                doc_lengths = doc_lengths.to(self.device)
                sent_lengths = sent_lengths.to(self.device)

                scores, word_att_weights, sentence_att_weights = self.model(docs, doc_lengths, sent_lengths)

                predictions = scores.max(dim=1)[1]
                correct_predictions = torch.eq(predictions, labels).sum().item()
                acc = correct_predictions

                self.accs.update(acc, batch_size)
            self.best_acc = max(self.best_acc, self.accs.avg)

            print('Test Average Accuracy: {acc.avg:.4f}'.format(acc=self.accs))
