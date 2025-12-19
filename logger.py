import torch
from collections import defaultdict
from datetime import datetime
from texttable import Texttable
import os
import numpy as np

class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) >= 4
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def get_best_epoch_index(self, epoch_metrics):

        valid_col = epoch_metrics[:, 1]
        max_valid = valid_col.max()
        candidate_indices = (valid_col == max_valid).nonzero().squeeze()


        if candidate_indices.numel() == 1:
            return candidate_indices.item()


        if epoch_metrics.size(1) >= 4:
            ood_col = epoch_metrics[:, 3]
            candidate_oods = ood_col[candidate_indices]
            max_ood = candidate_oods.max()
            best_candidate = candidate_indices[(candidate_oods == max_ood)]
            return best_candidate[0].item()
        return candidate_indices[0].item()

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = self.get_best_epoch_index(result)

            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'Highest In Test: {result[:, 2].max():.2f}')
            for i in range(result.size(1) - 3):
                print(f'Highest OOD Test: {result[:, i + 3].max():.2f}')
            print(f'Chosen epoch: {argmax + 1}')
            print(f'Final Train: {result[argmax, 0]:.2f}')
            print(f'Final In Test: {result[argmax, 2]:.2f}')
            for i in range(result.size(1) - 3):
                print(f'Final OOD Test: {result[argmax, i + 3]:.2f}')


        else:
            result = 100 * torch.tensor(self.results)
            best_results = []
            for r in result:
                train_high = r[:, 0].max().item()
                valid_high = r[:, 1].max().item()
                test_in_high = r[:, 2].max().item()
                test_ood_high = [r[:, i + 3].max().item() for i in range(r.size(1) - 3)]


                best_epoch_idx = self.get_best_epoch_index(r)
                train_final = r[best_epoch_idx, 0].item()
                test_in_final = r[best_epoch_idx, 2].item()
                test_ood_final = [r[best_epoch_idx, i + 3].item() for i in range(r.size(1) - 3)]

                best_results.append([train_high, valid_high, test_in_high]
                                    + test_ood_high + [train_final, test_in_final]
                                    + test_ood_final)


            best_result = torch.tensor(best_results)
            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'Highest In Test: {r.mean():.2f} ± {r.std():.2f}')
            ood_size = result[0].size(1)-3
            for i in range(ood_size):
                r = best_result[:, i+3]
                print(f'Highest OOD Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, ood_size+3]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, ood_size+4]
            print(f'   Final In  Test: {r.mean():.2f} ± {r.std():.2f}')
            for i in range(ood_size):
                r = best_result[:, i+5+ood_size]
                print(f'   Final OOD Test: {r.mean():.2f} ± {r.std():.2f}')

    def output(self, args):
        result = 100 * torch.tensor(self.results)
        best_results = []
        for r in result:
            train_high = r[:, 0].max().item()
            valid_high = r[:, 1].max().item()
            test_in_high = r[:, 2].max().item()
            test_ood_high = [r[:, i + 3].max().item() for i in range(r.size(1) - 3)]


            best_epoch_idx = self.get_best_epoch_index(r)
            train_final = r[best_epoch_idx, 0].item()
            test_in_final = r[best_epoch_idx, 2].item()
            test_ood_final = [r[best_epoch_idx, i + 3].item() for i in range(r.size(1) - 3)]

            best_results.append([train_high, valid_high, test_in_high]
                                + test_ood_high + [train_final, test_in_final]
                                + test_ood_final)


        best_result = torch.tensor(best_results)
        _dict = vars(args)
        table = Texttable()
        table.add_row(["Parameter", "Value"])
        for k in _dict:
            table.add_row([k, _dict[k]])

        if not os.path.exists(f'results/{args.dataset}'):
            os.makedirs(f'results/{args.dataset}/')
        datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f'results/{args.dataset}/lr_{args.lr}.wd_{args.weight_decay}.tau_{args.tau}.K_{args.K}.dp_{args.dropout}.{datetime_now}.txt'
        with open(f"{filename}", 'a') as f:
            f.write(table.draw())
            f.write(f'\nAll runs:\n')
            r = best_result[:, 0]
            f.write(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}\n')
            r = best_result[:, 1]
            f.write(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}\n')
            r = best_result[:, 2]
            f.write(f'Highest In Test: {r.mean():.2f} ± {r.std():.2f}\n')
            ood_size = result[0].size(1)-3
            for i in range(ood_size):
                r = best_result[:, i+3]
                f.write(f'Highest OOD Test: {r.mean():.2f} ± {r.std():.2f}\n')
            r = best_result[:, ood_size+3]
            f.write(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}\n')
            r = best_result[:, ood_size+4]
            f.write(f'   Final In  Test: {r.mean():.2f} ± {r.std():.2f}\n')
            for i in range(ood_size):
                r = best_result[:, i+5+ood_size]
                f.write(f'   Final OOD Test: {r.mean():.2f} ± {r.std():.2f}\n')
