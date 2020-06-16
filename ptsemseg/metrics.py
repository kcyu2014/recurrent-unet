# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
import IPython
import numpy as np
# from sklearn import metrics

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class runningScore(object):
    def __init__(self, n_classes, void=False):
        self.n_classes = n_classes
        self.last_void_class = void
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self._is_usable = False
        # self.label_trues = []
        # self.label_preds = []
        self.break_even_threshold = 100
        # self.tp = np.zeros(self.break_even_threshold, np.int)
        # self.fn = np.zeros(self.break_even_threshold, np.int)
        # self.fp = np.zeros(self.break_even_threshold, np.int)
        self.confusion_matrix_at_threshold = np.zeros((self.break_even_threshold, 2, 2))
        self.break_even = 0.
        self.n_lp = []
        self.n_lt = []

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def compute_break_even(self):
        """
        Faster way to compute such thing.
        :return:
        """
        assert len(self.n_lp) > 0 and len(self.n_lp) == len(self.n_lt)
        assert np.all([l.shape == p.shape for l, p in zip(self.n_lt, self.n_lp)])

        # total_lp = np.concatenate([np.expand_dims(l, axis=0) for l in self.n_lp])
        # total_lt = np.concatenate([np.expand_dims(l, axis=0) for l in self.n_lt])
        total_lp = np.concatenate(self.n_lp)
        total_lp = (total_lp * self.break_even_threshold).astype(np.int)
        total_lp = total_lp.astype(np.float) / self.break_even_threshold
        total_lt = np.concatenate(self.n_lt)

        # metrics.precision_recall_curve()
        # complete_lp = np.concatenate(self.break_even_threshold * [np.expand_dims(total_lp, axis=0)], axis=0)
        # cond = np.arange(0, self.break_even_threshold).astype(np.float) / self.break_even_threshold
        # cond = np.expand_dims(cond, axis=1)
        # filter_lp = np.where(complete_lp < cond, np.zeros(complete_lp), np.ones_like(complete_lp))

        def _compute_prec_recall(confusion_matrix):
            hist = confusion_matrix
            acc_cls = np.diag(hist) / hist.sum(axis=1)
            acc_cls = np.nanmean(acc_cls)
            recall = np.diag(hist) / hist.sum(axis=0)
            recall = np.nanmean(recall)
            return acc_cls, recall

        def _cut_from_threshold(in_tensor, th):
            t = np.copy(in_tensor)
            t[t < th] = 0
            t[t != 0] = 1
            t = t.astype(np.int)
            return t

        def _binary_search_split(start, end):
            # break condition
            if start + 1 == end or start == end:

                th = float(start / self.break_even_threshold)
                p, r = _compute_prec_recall(
                    self._fast_hist(
                        total_lt,
                        _cut_from_threshold(total_lp, th),
                        2
                    )
                )
                print("End point reach of binary search. found p {} r {} at {}".format(p, r, th))
                return p, r, th
            curr = (start + end) // 2
            th = float(curr) / self.break_even_threshold
            p, r = _compute_prec_recall(
                self._fast_hist(
                    total_lt,
                    _cut_from_threshold(total_lp, th),
                    2
                )
            )
            if r < p:
                # print(f" r < p. go at {curr} , {end}")
                return _binary_search_split(curr, end)
            else:
                # print(f" r >= p. go at {start} , {curr}")
                return _binary_search_split(start, curr)

        prec, recall, thresh = _binary_search_split(0, self.break_even_threshold)
        print('prec {}, recall {}'.format(prec, recall))
        return prec, recall, thresh

    def compute_break_even_slow(self):
        # precision, recall, threshold = metrics.precision_recall_curve(self.label_trues, self.label_preds)
        # = metrics.confusion_matrix()
        # return precision, recall, threshold
        # precision_by_threshold = self.tp / (self.fn

        recalls = np.zeros(self.break_even_threshold, np.float)
        precs = np.zeros(self.break_even_threshold, np.float)

        crossing = 0.
        num_crossing = 0
        for i in range(self.break_even_threshold):
            hist = self.confusion_matrix_at_threshold[i, :, :]
            acc_cls = np.diag(hist) / hist.sum(axis=1)
            acc_cls = np.nanmean(acc_cls)
            recall = np.diag(hist) / hist.sum(axis=0)
            recall = np.nanmean(recall)
            recalls[i] = recall
            precs[i] = acc_cls
            if recall < acc_cls:
                crossing = i
                num_crossing += 1

        # Find the closest splitting point.
        return float(crossing / self.break_even_threshold), num_crossing

    def update(self, label_trues, label_preds, step=-1):
        self._is_usable = True
        for lt, lp in zip(label_trues, label_preds):
            if self.last_void_class:
                lp = lp[lt < self.n_classes -1]
                lt = lt[lt < self.n_classes -1]
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def update_raw_slow(self, label_trues, label_preds, step=-1):

        assert self.n_classes == 2 or self.n_classes == 3 and self.last_void_class
        if isinstance(label_trues, list) and len(label_trues) > 0:
            label_trues = label_trues[-1]
            label_preds = label_preds[-1]

        # only update the
        # if step == -1 or step == 2:
        #     self.label_trues.extend(label_trues)
        #     self.label_preds.extend(label_preds)

        def _cut_from_threshold(t, th):
            t = softmax(t)
            t = t[1,:]
            t[t < th] = 0
            t[t != 0] = 1
            t = t.astype(np.int)
            return t

        for i in range(self.break_even_threshold):
            for lt, lp in zip(label_trues, label_preds):
                # such only support for binary
                # remove this -1 only take the first two classes
                n_lp = lp[:2,lt < 2]
                n_lt = lt[lt < 2]

                n_lp_p = _cut_from_threshold(n_lp, th=float(i / self.break_even_threshold)).flatten()

                self.confusion_matrix_at_threshold[i] += self._fast_hist(
                    n_lt,
                    n_lp_p,
                    n_class=2
                )
        # IPython.embed()

    def update_raw(self, label_trues, label_preds, step=-1):
        if step == -1 or step == 2:
            pass
        else:
            return

        assert self.n_classes == 2 or self.n_classes == 3 and self.last_void_class
        if isinstance(label_trues, list) and len(label_trues) > 0:
            label_trues = label_trues[-1]
            label_preds = label_preds[-1]

        for lt, lp in zip(label_trues, label_preds):
            # such only support for binary
            # remove this -1 only take the first two classes
            n_lp = lp[:2, lt < 2]
            n_lt = lt[lt < 2]
            n_lp = softmax(n_lp)[1, :]
            self.n_lp.append(n_lp)
            self.n_lt.append(n_lt)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
            - AUC and the P-R break even point.
        """

        if self.last_void_class:
            hist = self.confusion_matrix[:-1, :-1]
        else:
            hist = self.confusion_matrix
        # IPython.embed()
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        prec = acc_cls.copy()
        acc_harmonic = self.n_classes/(1/acc_cls).sum()
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        recall = np.diag(hist) / hist.sum(axis=0)
        mean_recall = np.nanmean(recall)
        f1score = 2/(1/acc_cls + 1/mean_recall)
        f1 = 2/(1/prec + 1/recall)
        cls_f1 = dict(zip(range(self.n_classes), f1))
        return (
            {
                "Overall Acc: \t": acc,
                "Mean Prec : \t": acc_cls,
                "FreqW Prec : \t": fwavacc,
                "Harm Prec: \t": acc_harmonic,
                "Mean IoU : \t": mean_iu,
                "Mean Recall : \t": mean_recall,
                "Mean f1score : \t": f1score
            },
            cls_iu,
            cls_f1,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

