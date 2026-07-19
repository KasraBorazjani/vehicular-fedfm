import torch
import copy
from prettytable import PrettyTable
import numpy as np
from torch.nn import functional as F
from typing import Dict, List, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Federated aggregation helpers
# ══════════════════════════════════════════════════════════════════════════════

def fedavg_aggregate(
    local_states: List[Dict[str, torch.Tensor]],
    weights:      Optional[List[float]] = None,
    keys:         Optional[List[str]]   = None,
) -> Dict[str, torch.Tensor]:
    """
    Data-weighted FedAvg over a list of state_dicts.

    Args:
        local_states : per-client state_dicts (CPU tensors).
        weights      : per-client weights (proportional to dataset size).
                       Uniform if None.
        keys         : subset of keys to aggregate; all keys if None.
    Returns:
        Aggregated state dict.
    """
    if not local_states:
        raise ValueError("fedavg_aggregate received empty local_states")

    if weights is None:
        weights = [1.0 / len(local_states)] * len(local_states)
    total  = sum(weights)
    w_norm = [w / total for w in weights]

    ref_keys = keys if keys is not None else list(local_states[0].keys())
    agg: Dict[str, torch.Tensor] = {}
    for k in ref_keys:
        if k not in local_states[0]:
            continue
        dtype = local_states[0][k].dtype
        acc   = local_states[0][k].float() * w_norm[0]
        for w, s in zip(w_norm[1:], local_states[1:]):
            if k in s:
                acc = acc + s[k].float() * w
        agg[k] = acc.to(dtype)
    return agg


def fedmma_aggregate(
    local_states: List[Dict[str, torch.Tensor]],
    global_keys:  List[str],
    weights:      Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    FedMMA selective aggregation: only average keys in global_keys.
    Task-head (local) keys are omitted — callers merge them per-client.
    """
    return fedavg_aggregate(local_states, weights, keys=global_keys)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_miou(
    preds:        torch.Tensor,
    labels:       torch.Tensor,
    num_classes:  int,
    ignore_index: int = 255,
) -> Dict[str, float]:
    """
    Mean Intersection-over-Union for semantic / 3-D segmentation.

    Args:
        preds  : [N, H, W] int (argmax) or [N, C, H, W] logits.
        labels : [N, H, W] int ground-truth.
    Returns:
        {"miou": float, "per_class_iou": List[float|nan]}
    """
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)

    preds  = preds.view(-1).long()
    labels = labels.view(-1).long()
    valid  = labels != ignore_index
    preds, labels = preds[valid], labels[valid]

    ious: List[float] = []
    for c in range(num_classes):
        pred_c = preds  == c
        true_c = labels == c
        tp     = (pred_c & true_c).sum().item()
        fp     = (pred_c & ~true_c).sum().item()
        fn     = (~pred_c & true_c).sum().item()
        denom  = tp + fp + fn
        ious.append(float(tp) / denom if denom > 0 else float("nan"))

    valid_ious = [x for x in ious if not np.isnan(x)]
    miou       = float(np.mean(valid_ious)) if valid_ious else 0.0
    return {"miou": miou, "per_class_iou": ious}


def _compute_ap(recalls: List[float], precisions: List[float]) -> float:
    """11-point interpolated average precision."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        candidates = [p for r, p in zip(recalls, precisions) if r >= t]
        ap += max(candidates) if candidates else 0.0
    return ap / 11.0


def compute_map50(
    pred_boxes_list:  List[torch.Tensor],
    pred_scores_list: List[torch.Tensor],
    pred_labels_list: List[torch.Tensor],
    gt_boxes_list:    List[torch.Tensor],
    gt_labels_list:   List[torch.Tensor],
    num_classes:      int,
    iou_thresh:       float = 0.5,
) -> Dict[str, float]:
    """
    PASCAL-VOC style mAP@iou_thresh for 2-D detection.

    All box tensors in cxcywh normalised format [0, 1].
    Returns {"map50": float, "per_class_ap": List[float]}.
    """
    try:
        from torchvision.ops import box_iou, box_convert
    except ImportError:
        return {"map50": 0.0, "per_class_ap": [0.0] * num_classes}

    per_class_ap: List[float] = []
    for c in range(num_classes):
        all_scores: List[float] = []
        all_tp:     List[int]   = []
        all_fp:     List[int]   = []
        total_gt = 0

        for pb, sc, pl, gb, gl in zip(
            pred_boxes_list, pred_scores_list, pred_labels_list,
            gt_boxes_list,   gt_labels_list,
        ):
            pb_c = pb[pl == c]; sc_c = sc[pl == c]
            gb_c = gb[gl == c]
            total_gt += gb_c.shape[0]

            if pb_c.shape[0] == 0:
                continue

            order   = sc_c.argsort(descending=True)
            pb_c    = pb_c[order]
            sc_c    = sc_c[order]
            matched = torch.zeros(gb_c.shape[0], dtype=torch.bool)

            for i in range(pb_c.shape[0]):
                all_scores.append(sc_c[i].item())
                if gb_c.shape[0] == 0:
                    all_tp.append(0); all_fp.append(1)
                    continue
                bp       = box_convert(pb_c[i:i+1], "cxcywh", "xyxy")
                bg       = box_convert(gb_c, "cxcywh", "xyxy")
                iou_vals = box_iou(bp, bg)[0]
                best_j   = iou_vals.argmax().item()
                if iou_vals[best_j] >= iou_thresh and not matched[best_j]:
                    all_tp.append(1); all_fp.append(0)
                    matched[best_j] = True
                else:
                    all_tp.append(0); all_fp.append(1)

        if total_gt == 0:
            per_class_ap.append(0.0)
            continue

        ord_idx    = np.argsort([-s for s in all_scores])
        tp_sorted  = np.array(all_tp)[ord_idx]
        fp_sorted  = np.array(all_fp)[ord_idx]
        cum_tp     = np.cumsum(tp_sorted)
        cum_fp     = np.cumsum(fp_sorted)
        precisions = (cum_tp / (cum_tp + cum_fp + 1e-10)).tolist()
        recalls    = (cum_tp / (total_gt + 1e-10)).tolist()
        per_class_ap.append(_compute_ap(recalls, precisions))

    map50 = float(np.mean(per_class_ap)) if per_class_ap else 0.0
    return {"map50": map50, "per_class_ap": per_class_ap}


# ══════════════════════════════════════════════════════════════════════════════
# Multi-task gradient manipulation
# ══════════════════════════════════════════════════════════════════════════════

def cagrad(
    grads: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Conflict-Averse Gradient (Liu et al., NeurIPS 2021).

    Finds a gradient combination g* ≈ g_mean that minimises conflict
    (cosine angle between g* and each task gradient).

    Args:
        grads : [T, D] one row per task gradient (shared parameters only).
        alpha : blend coefficient.  0 → pure g_mean; 1 → pure g*.
    Returns:
        [D] combined gradient.
    """
    T, D   = grads.shape
    g_mean = grads.mean(0)
    if T == 1 or alpha == 0.0:
        return g_mean

    # Gram matrix
    GG = (grads @ grads.T).double()

    # Frank-Wolfe: minimise 0.5 c^T G c  s.t.  c ≥ 0, sum = 1
    c = torch.ones(T, dtype=torch.float64, device=grads.device) / T
    for i in range(30):
        gc    = c @ GG
        worst = gc.argmin()
        c_fw  = torch.zeros_like(c)
        c_fw[worst] = 1.0
        step  = 2.0 / (i + 2.0)
        c     = (1.0 - step) * c + step * c_fw

    c      = c.clamp(min=0).float()
    c      = c / c.sum()
    g_opt  = (c.unsqueeze(1) * grads).sum(0)
    scale  = g_mean.norm() / (g_opt.norm() + 1e-8)
    return alpha * g_opt * scale + (1.0 - alpha) * g_mean


class GradNorm:
    """
    GradNorm dynamic task-loss weighting (Chen et al., ICML 2018).

    Learns per-task weights w_t that normalise gradient magnitudes relative
    to each task's training rate.  Used in the `multitask` baseline only.

    Usage::

        gn = GradNorm(num_tasks=4, alpha=1.5, lr=1e-3)
        loss, w = gn.weighted_loss(task_losses)   # [T] → scalar, [T]
        loss.backward()
        gn.step(shared_param, task_losses.detach())
    """

    def __init__(self, num_tasks: int, alpha: float = 1.5, lr: float = 1e-3):
        self.alpha     = alpha
        self.num_tasks = num_tasks
        self.weights   = torch.ones(num_tasks, requires_grad=True, dtype=torch.float32)
        self._optim    = torch.optim.Adam([self.weights], lr=lr)
        self._L0: Optional[torch.Tensor] = None

    def weighted_loss(
        self, task_losses: torch.Tensor
    ):
        """
        Return weighted sum and the softmax-normalised weight vector.

        Args:
            task_losses : [T] per-task losses (attached to graph).
        """
        if self._L0 is None:
            self._L0 = task_losses.detach().clone()
        w     = torch.softmax(self.weights.to(task_losses.device), dim=0) * self.num_tasks
        total = (w * task_losses).sum()
        return total, w

    def step(
        self,
        shared_param: torch.Tensor,
        task_losses:  torch.Tensor,
    ) -> None:
        """
        Update weight parameters via the GradNorm auxiliary loss.

        Args:
            shared_param : any shared-backbone parameter tensor (for ref norm).
            task_losses  : [T] detached per-task losses from the current step.
        """
        if self._L0 is None or shared_param.grad is None:
            return
        self._optim.zero_grad()
        w   = torch.softmax(self.weights, dim=0) * self.num_tasks
        G_W = shared_param.grad.norm().detach()
        r_t = (task_losses.detach() / (self._L0.to(task_losses.device) + 1e-8))
        r_t = r_t / (r_t.mean() + 1e-8)
        gn_loss = (w * G_W - G_W * r_t.pow(self.alpha)).abs().sum()
        gn_loss.backward()
        self._optim.step()
        self.weights.data.clamp_(min=0.01)

def show_results(cfg, results, epoch,global_test_acc_dict):

    global_test_acc = []
    global_test_error = []
    global_test_f1 = []
    for k, result in enumerate(results):
        global_test_acc.append(results[k]['accuracy'])
        global_test_error.append(results[k]['error_rate'])
        global_test_f1.append(results[k]['macro_f1'])

        if k in global_test_acc_dict:
            global_test_acc_dict[k].append(results[k]['accuracy'])
        else:
            global_test_acc_dict[k] = [results[k]['accuracy']]

        print(k, "--Local test acc:", results[k]['accuracy'])

    print("--Global test acc:", sum(global_test_acc) / len(global_test_acc))

    print(f"Epoch:{epoch}")
    return global_test_acc,global_test_acc_dict

def average_weights(w, idxs_users, datanumber_client, islist=False):
    """
    Returns the average of the weights.
    """
    total_data_points = sum([datanumber_client[r] for r in idxs_users])

    w_avg = copy.deepcopy(w[idxs_users[0]])
    for idx in range(len(idxs_users)):
        fed_avg_freqs = datanumber_client[idxs_users[idx]] / total_data_points

        if islist:
            if idx == 0:
                w_avg = w_avg * fed_avg_freqs
            else:
                w_avg += w[idxs_users[idx]] * fed_avg_freqs
        else:
            if idx == 0:
                for key in w_avg:
                    w_avg[key] = w_avg[key] * fed_avg_freqs
            else:
                for key in w_avg:
                    w_avg[key] += w[idxs_users[idx]][key] * fed_avg_freqs

    return w_avg

def aggregate_local_models(representation,datanumber_client):
    users=range(len(representation))

    total_data = sum(datanumber_client[u] for u in users)
    

    first = representation[users[0]]
    agg_global = {k: torch.zeros_like(v) for k, v in first.items()}
    for u in users:
        w = float(datanumber_client[u])/float(total_data)
        for k, v in representation[u].items():
            agg_global[k].add_(v * w)

    return agg_global
    

def moment_aggre_weights(w_prompt,w_key, global_key, global_group_prompt,cluster_size ,idxs_users, datanumber_client, moment = 0.5):
    
    total_data_points = sum([datanumber_client[r] for r in idxs_users])
    total_cluster_size = sum([cluster_size[r] for r in idxs_users])

    group_ratio = [[0 for r in range(global_key.shape[0])] for i in idxs_users]

    for idx in range(len(idxs_users)):
        for r in range(global_key.shape[0]):
            if total_cluster_size[r] <= 50:
                group_ratio[idx][r] = 1/len(idxs_users)
            else:
                group_ratio[idx][r] =  (cluster_size[idx][r] / total_cluster_size[[r]]).item()

    for idx in range(len(idxs_users)):
        fed_avg_freqs = datanumber_client[idxs_users[idx]] / total_data_points

        if idx == 0:
            unique_key = copy.deepcopy(global_key)
            unique_group_prompt = copy.deepcopy(global_group_prompt)
            global_group_prompt = w_prompt[idx] * fed_avg_freqs
            for i, s in enumerate(group_ratio[idx]):
                global_key[i] = w_key[idx][i] * s
        else:
            global_group_prompt += w_prompt[idx] * fed_avg_freqs
            for i, s in enumerate(group_ratio[idx]):
                global_key[i] += w_key[idx][i] * s

    moment_key_para = moment * unique_key + (1 - moment) * global_key
    moment_prompt_para = moment * unique_group_prompt + (1 - moment) * global_group_prompt
    return moment_prompt_para, moment_key_para


def count_parameters(model, model_name):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if model_name in name:
            # if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params



import os

def save_acc_csv(para_dir,global_test_acc_dict,cfg):
    acc_path = os.path.join(para_dir, 'acc.csv')
    if os.path.exists(acc_path):
        with open(acc_path, 'a') as result_file:
            for key in global_test_acc_dict:
                method_result = global_test_acc_dict[key]
                result_file.write(str(key) + ',')
                for epoch in range(len(method_result)):
                    result_file.write(str(method_result[epoch]))
                    if epoch != len(method_result) - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')
    else:
        with open(acc_path, 'w') as result_file:
            result_file.write('idx,')
            for epoch in range(cfg.OPTIM.ROUND):
                result_file.write('epoch_' + str(epoch))
                if epoch != cfg.OPTIM.ROUND - 1:
                    result_file.write(',')
                else:
                    result_file.write('\n')

            for key in global_test_acc_dict:
                method_result = global_test_acc_dict[key]
                result_file.write(str(key) + ',')
                for epoch in range(len(method_result)):
                    result_file.write(str(method_result[epoch]))
                    if epoch != len(method_result) - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')
            
   
class KMEANS:
    def __init__(self, n_clusters=2, max_iter=50, verbose=True,device = torch.device("cpu")):
        self.n_cluster = n_clusters
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.verbose = verbose
        self.started = False
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit_predict(self, x):
        # 随机选择初始中心点
        init_row = torch.randperm(x.shape[0])[:self.n_clusters].to(self.device)
        init_points = x[init_row]
        self.centers = init_points
        while True:
            # 聚类标记
            self.nearest_center(x)
            # 更新中心点
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmax(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 or self.max_iter == self.count:
                break
            self.count += 1

        return self.labels

    def nearest_center(self, x): 
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        for i, sample in enumerate(x):
            dist = F.cosine_similarity(sample.unsqueeze(0), self.centers,dim=1)
            labels[i] = torch.argmax(dist)
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
        self.labels = labels
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers
