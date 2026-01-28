import math
import os
import json
import time
import torch
import numpy as np
import random
import torch.nn.functional as F
import torchvision.utils


class SpliceMix(object):
    def __init__(self, mode='SpliceMix', grids=('2x2',), n_grids=(0,), mix_prob=1.):
        # mode: 'SpliceMix' for custom grid setting; 'SpliceMix--Default=True' for default setting; 'SpliceMix--Mini=True' for minimalism setting
        # grids: grid settings, e.g., ['1x2', '2x2', '2x3-3']
        # n_grids: number of mixed samples in each setting, e.g., [3, 2, 1]
        # mix_prob: probability of using SpliceMix per mini-batch

        super(SpliceMix, self).__init__()

        self.Default = False
        self.Mini = False
        self.checkMode(mode)

        self.mix_prob = mix_prob
        self.grids = grids
        self.n_grids = n_grids
        self.use_asym = True
        self.config_default = {'1x2': .7, '2x2': .3, '2x3': .0, 'drop_rate': .3}

        # Module 1: non-gradient saliency (feature-free) to guide donor selection and dropping.
        self.use_saliency = True  # switch for saliency-guided selection
        self.saliency_weights = {'norm': 0.5, 'edge': 0.5}
        self.saliency_top_p = 0.6  # base top-p; will be adapted by label density
        self.saliency_top_p_min = 0.5
        self.saliency_top_p_max = 0.75
        self.saliency_var_threshold = 1e-4  # fallback to random when saliency is near-constant
        self.saliency_smooth_kernel = 3
        self.saliency_scales = [1.0, 0.5, 0.25]
        self.saliency_eps = 1e-6
        # group-aware mixing based on offline co-occurrence stats
        self.group_mix_enable = True
        self.group_meta_path = os.path.join(os.path.dirname(__file__), 'cooccur_voc2007.json')
        self.group_grids_related = ['1x2', '2x2']
        self.group_grids_diverse = ['1x2', '2x2']
        self.group_top_p_scale_related = 1.0
        self.group_top_p_scale_diverse = 1.0
        self.group_log_prob = 0.02  # probability to print group stats per call
        self.compat_log_prob = 0.01  # probability to log label-compatible pairing stats
        self.pmi_threshold = -1.0  # PMI threshold for label compatibility (lower = more partners)
        
        # CO/DC dual-branch mixing strategy (inspired by correlative-discriminative balance)
        self.dual_branch_enable = True
        self.dc_base_prob = 0.3  # base probability to use DC (discriminative) branch
        self.dc_adaptive = True  # dynamically adjust DC prob based on label density
        self.dc_density_scale = 0.5  # how much label density affects DC prob
        
        # V2: Tiered Rare-Class Aware Mixing (分档稀有类感知)
        self.rare_class_aware = True
        self.use_percentile_threshold = True  # 使用分位数阈值
        
        # 分档阈值 (loaded from cooccur file)
        self.rare_threshold_p10 = 0  # 极稀有阈值 (P10)
        self.rare_threshold_p25 = 0  # 稀有阈值 (P25)
        self.rare_classes_p10 = set()  # 极稀有类索引
        self.rare_classes_p25 = set()  # 稀有类索引
        
        # 分档DC缩放因子
        self.dc_reduction_very_rare = 0.50  # 极稀有: DC * 0.3
        self.dc_reduction_rare = 0.80       # 稀有: DC * 0.6
        
        # Fallback for old behavior
        self.rare_class_threshold = 200
        self.rare_class_dc_reduction = 0.5
        # Curriculum Learning: DC probability increases over training
        # Early: focus on CO (learn label co-occurrence patterns)
        # Late: focus on DC (enhance discriminative features, prevent overfitting)
        self.dc_curriculum_enable = True
        self.dc_curriculum_min = 0.1   # DC prob at epoch 0
        self.dc_curriculum_max = 0.70   # DC prob at final epoch
        self.current_epoch = 0
        self.total_epochs = 80
        
        # Curriculum Learning: DC probability increases over training
        # Early: focus on CO (learn label co-occurrence patterns)
        # Late: focus on DC (enhance discriminative features, prevent overfitting)
        self.dc_curriculum_enable = True
        self.dc_curriculum_min = 0.1   # DC prob at epoch 0
        self.dc_curriculum_max = 0.70   # DC prob at final epoch
        self.current_epoch = 0
        self.total_epochs = 80
        # ===== Module 3: Curriculum Diversity Mixing (CDM) =====
        # 创新点：训练早期用相似图像混合，后期用多样图像混合
        self.cdm_enable = True
        self.cdm_diverse_prob_min = 0.05   # 早期 diverse 概率
        self.cdm_diverse_prob_max = 0.25   # 后期 diverse 概率
        self.cdm_warmup_epochs = 30      # 前N个epoch强制related模式
        self.cdm_cosine_schedule = True   # 使用cosine调度替代线性
        self.cdm_rare_protection = True   # 稀有类样本降低diverse概率
        self.cdm_rare_diverse_scale = 0.5 # 稀有类的diverse概率缩放因子
        # ===== Module 2: CAM-Guided Part-Level Mix =====
        self.cam_guided_enable = False  # V2: 启用 CAM-Guided
        self.cam_mixer = None  # 延迟初始化
        
        # optional visualization of mixed outputs
        self.viz_enable = True
        self.viz_prob = 0.02
        self.viz_max = 20
        self.viz_dir = os.path.join(os.path.dirname(__file__), 'visualization')
        self.viz_count = 0

        # entropy-aware scoring + temperature for top-p selection
        self.saliency_entropy_weight = 0.2
        self.saliency_temp_min = 0.7
        self.saliency_temp_max = 1.5

        self.group_meta = self._load_group_meta(self.group_meta_path) if self.group_mix_enable else {}
        self.mixer = self.Smix if self.Mini == False else self.Smix_minimalism
        if self.Default: print("SpliceMix w/ default setting")
        if self.Mini: print("SpliceMix w/ minimalism setting")


    def _load_group_meta(self, path):
        try:
            if not os.path.exists(path):
                return {}
            with open(path, 'r') as f:
                meta = json.load(f)
            meta['related'] = meta.get('related', [])
            # Store class frequency for rare-class aware mixing
            if 'freq' in meta:
                self.class_freq = meta['freq']
            
            # V2: Load percentile-based thresholds
            if self.use_percentile_threshold:
                if "rare_p10" in meta and "rare_p25" in meta:
                    self.rare_threshold_p10 = meta["rare_p10"]
                    self.rare_threshold_p25 = meta["rare_p25"]
                    self.rare_classes_p10 = set(meta.get("rare_classes_p10", []))
                    self.rare_classes_p25 = set(meta.get("rare_classes_p25", []))
                    print(f"[SpliceMix V2] Tiered: P10={self.rare_threshold_p10:.0f}, P25={self.rare_threshold_p25:.0f}")
                    print(f"[SpliceMix V2] 极稀有类 ({len(self.rare_classes_p10)}), 稀有类 ({len(self.rare_classes_p25)})")
            meta['discriminative'] = meta.get('discriminative', [])
            
            # Compute PMI-based strong pairs (generic strategy for high-freq class issue)
            freq = meta.get('freq', [])
            cooccur = meta.get('cooccurrence', [])
            if freq and cooccur:
                meta['pmi_adj'] = self._compute_pmi_adjacency(freq, cooccur, pmi_threshold=self.pmi_threshold)
            
            return meta
        except Exception as e:
            print(f"[SpliceMix] warn: failed to load group meta {path}: {e}")
            return {}

    def _compute_pmi_adjacency(self, freq, cooccur, pmi_threshold=0.5):
        """
        Compute PMI-based adjacency matrix for label compatibility.
        PMI(i,j) = log(P(i,j) / (P(i) * P(j)))
        
        High PMI = co-occurrence exceeds random expectation = true semantic relation
        This naturally down-weights high-frequency classes like 'person'.
        
        Args:
            freq: list of per-class frequencies
            cooccur: co-occurrence matrix
            pmi_threshold: minimum PMI to consider as compatible
        
        Returns:
            dict mapping class_idx -> list of compatible class indices
        """
        import numpy as np
        freq = np.array(freq)
        cooccur = np.array(cooccur)
        C = len(freq)
        total = freq.sum()
        
        # P(i) = freq[i] / total
        p_i = freq / total
        
        # PMI matrix
        pmi = np.zeros((C, C))
        for i in range(C):
            for j in range(C):
                if i == j:
                    pmi[i, j] = float('inf')  # same class always compatible
                    continue
                p_ij = cooccur[i][j] / total if total > 0 else 0
                p_i_j = p_i[i] * p_i[j]
                if p_ij > 0 and p_i_j > 0:
                    pmi[i, j] = np.log(p_ij / p_i_j + 1e-10)
                else:
                    pmi[i, j] = -float('inf')
        
        # Build adjacency based on PMI threshold
        pmi_adj = {}
        for i in range(C):
            partners = []
            for j in range(C):
                if i != j and pmi[i, j] > pmi_threshold:
                    partners.append(j)
            pmi_adj[str(i)] = partners
        
        # Log stats
        avg_partners = sum(len(v) for v in pmi_adj.values()) / len(pmi_adj) if pmi_adj else 0
        print(f"[SpliceMix] PMI adjacency: threshold={pmi_threshold}, avg_partners={avg_partners:.1f}")
        
        return pmi_adj

    def _build_cooccur_adj(self, num_classes, device):
        """Build co-occurrence adjacency matrix from offline stats."""
        # adj[i,j] = True means class i and j are compatible (can be mixed)
        adj = torch.eye(num_classes, dtype=torch.bool, device=device)  # same class always compatible
        
        if not self.group_meta:
            return adj
        
        # Prefer PMI-based adjacency (handles high-freq class issue generically)
        pmi_adj = self.group_meta.get('pmi_adj', {})
        if pmi_adj:
            for k, v in pmi_adj.items():
                i = int(k)
                for j in v:
                    if 0 <= i < num_classes and 0 <= j < num_classes:
                        adj[i, j] = True
                        adj[j, i] = True  # symmetric
            return adj
        
        # Fallback to strong co-occurrence pairs
        strong_per_class = self.group_meta.get('strong_per_class', {})
        if strong_per_class:
            for k, v in strong_per_class.items():
                i = int(k)
                for j in v:
                    if 0 <= i < num_classes and 0 <= j < num_classes:
                        adj[i, j] = True
                        adj[j, i] = True  # symmetric
        else:
            strong_pairs = self.group_meta.get('strong_pairs', [])
            for i, j in strong_pairs:
                if 0 <= i < num_classes and 0 <= j < num_classes:
                    adj[i, j] = True
                    adj[j, i] = True
        
        return adj

    def _select_compatible_donors(self, targets, anchor_idx, num_donors, cooccur_adj, used_set=None):
        """
        Select donors that are label-compatible with the anchor.
        
        Args:
            targets: [B, C] label tensor
            anchor_idx: index of anchor sample
            num_donors: number of donors needed
            cooccur_adj: [C, C] co-occurrence adjacency matrix
            used_set: set of already used indices to avoid
        
        Returns:
            list of donor indices (guaranteed to be within [0, B-1])
        """
        if used_set is None:
            used_set = set()
        
        B, C = targets.shape
        anchor_labels = (targets[anchor_idx] > 0).float()  # [C]
        
        # Method 1: Find samples sharing at least one positive label
        label_pos = (targets > 0).float()  # [B, C]
        shared_labels = (label_pos @ anchor_labels) > 0  # [B]
        
        # Method 2: Find samples with co-occurring classes
        compatible_classes = ((cooccur_adj.float() @ anchor_labels) > 0).float()  # [C]
        cooccur_compat = (label_pos @ compatible_classes) > 0  # [B]
        
        # Combine: either shared label OR co-occurrence compatible
        compatible = shared_labels | cooccur_compat  # [B]
        
        # Build candidate list (exclude anchor and used)
        all_used = used_set | {anchor_idx}
        candidates = []
        fallback_pool = []
        
        for i in range(B):
            if i in all_used:
                continue
            if compatible[i]:
                candidates.append(i)
            else:
                fallback_pool.append(i)
        
        # Select donors: prioritize compatible, then fallback
        selected = []
        if len(candidates) >= num_donors:
            selected = random.sample(candidates, num_donors)
        else:
            selected = candidates[:]
            need = num_donors - len(selected)
            if need > 0 and len(fallback_pool) > 0:
                take = min(need, len(fallback_pool))
                selected.extend(random.sample(fallback_pool, take))
                need -= take
            # Still not enough? Reuse from all except anchor
            if need > 0:
                reuse_pool = [i for i in range(B) if i != anchor_idx and i not in selected]
                if len(reuse_pool) > 0:
                    take = min(need, len(reuse_pool))
                    selected.extend(random.sample(reuse_pool, take))
        
        # Record stats for logging
        self._last_compat_stats = {
            'anchor_idx': anchor_idx,
            'num_candidates': len(candidates),
            'num_fallback': max(0, num_donors - len(candidates)),
            'total_donors': num_donors
        }
        
        # Ensure we have exactly num_donors (pad with anchor if absolutely necessary)
        while len(selected) < num_donors:
            # Last resort: duplicate from what we have or use anchor
            if len(selected) > 0:
                selected.append(random.choice(selected))
            else:
                selected.append(anchor_idx)
        
        return selected[:num_donors]

    def _select_discriminative_donors(self, targets, anchor_idx, num_donors, cooccur_adj, used_set=None):
        """
        DC Branch: Semi-Discriminative Strategy (Improved)
        
        Instead of selecting completely unrelated samples (too extreme, may cause noise),
        we select samples with PARTIAL overlap: share 1-2 labels but also have different ones.
        
        This is similar to "hard negatives" in contrastive learning:
        - Partial overlap provides semantic anchor (stability)
        - Different labels force discriminative learning (challenge)
        
        Priority:
        1. Best: 1-2 shared labels AND has unique labels (hard negative)
        2. Good: 3+ shared labels but also has unique labels  
        3. Fallback: completely different (original DC)
        4. Last resort: fully compatible
        
        Args:
            targets: [B, C] label tensor
            anchor_idx: index of anchor sample
            num_donors: number of donors needed
            cooccur_adj: [C, C] co-occurrence adjacency matrix
            used_set: set of already used indices to avoid
        
        Returns:
            list of donor indices
        """
        import random
        if used_set is None:
            used_set = set()
        
        B, C = targets.shape
        anchor_labels = (targets[anchor_idx] > 0).float()  # [C]
        anchor_count = anchor_labels.sum().item()
        
        label_pos = (targets > 0).float()  # [B, C]
        sample_counts = label_pos.sum(dim=1)  # [B] - number of labels per sample
        
        # Compute shared label count for each sample
        shared_count = (label_pos * anchor_labels).sum(dim=1)  # [B]
        
        # Compute unique labels (labels in sample but not in anchor)
        unique_in_sample = ((label_pos - anchor_labels) > 0).float().sum(dim=1)  # [B]
        
        # Build candidate pools by semi-discriminative priority
        all_used = used_set | {anchor_idx}
        pool_hard_neg = []    # 1-2 shared + has unique (ideal hard negative)
        pool_soft_neg = []    # 3+ shared + has unique (softer)
        pool_full_disc = []   # 0 shared (original DC, more extreme)
        pool_fallback = []    # fully compatible (no unique labels)
        
        for i in range(B):
            if i in all_used:
                continue
            shared = shared_count[i].item()
            unique = unique_in_sample[i].item()
            
            if shared >= 1 and shared <= 2 and unique >= 1:
                # Best: partial overlap with discriminative potential
                pool_hard_neg.append((i, unique))  # store unique count for scoring
            elif shared >= 3 and unique >= 1:
                # Good: more overlap but still has unique labels
                pool_soft_neg.append((i, unique))
            elif shared == 0:
                # Original DC: completely different
                pool_full_disc.append(i)
            else:
                # Fallback: compatible (shared > 0, unique = 0)
                pool_fallback.append(i)
        
        # Sort hard/soft negatives by unique count (prefer more unique = harder)
        pool_hard_neg.sort(key=lambda x: x[1], reverse=True)
        pool_soft_neg.sort(key=lambda x: x[1], reverse=True)
        
        # Select donors with priority
        selected = []
        
        # Priority 1: hard negatives (best for discriminative learning)
        for item in pool_hard_neg:
            if len(selected) >= num_donors:
                break
            selected.append(item[0])
        
        # Priority 2: soft negatives
        for item in pool_soft_neg:
            if len(selected) >= num_donors:
                break
            selected.append(item[0])
        
        # Priority 3: fully discriminative (original DC)
        if len(selected) < num_donors and len(pool_full_disc) > 0:
            need = num_donors - len(selected)
            take = min(need, len(pool_full_disc))
            selected.extend(random.sample(pool_full_disc, take))
        
        # Priority 4: fallback
        if len(selected) < num_donors and len(pool_fallback) > 0:
            need = num_donors - len(selected)
            take = min(need, len(pool_fallback))
            selected.extend(random.sample(pool_fallback, take))
        
        # Still not enough? Reuse from all except anchor
        if len(selected) < num_donors:
            reuse_pool = [i for i in range(B) if i != anchor_idx and i not in selected]
            if len(reuse_pool) > 0:
                take = min(num_donors - len(selected), len(reuse_pool))
                selected.extend(random.sample(reuse_pool, take))
        
        # Record stats
        self._last_dc_stats = {
            'anchor_idx': anchor_idx,
            'pool_hard_neg': len(pool_hard_neg),
            'pool_soft_neg': len(pool_soft_neg),
            'pool_full_disc': len(pool_full_disc),
            'pool_fallback': len(pool_fallback),
            'total_donors': num_donors
        }
        
        # Pad if absolutely necessary
        while len(selected) < num_donors:
            if len(selected) > 0:
                selected.append(random.choice(selected))
            else:
                selected.append(anchor_idx)
        
        return selected[:num_donors]

    def _compute_relevance(self, targets, anchor_idx, donor_indices, is_dc_branch=False):
        """
        Compute relevance scores between anchor and each donor.
        
        For CO branch: higher Jaccard similarity = higher relevance
        For DC branch: "hard negative quality" - 1-2 shared + unique labels is optimal
        """
        anchor_labels = (targets[anchor_idx] > 0).float()
        anchor_count = anchor_labels.sum().item()
        
        relevance_scores = []
        
        for donor_idx in donor_indices:
            donor_labels = (targets[donor_idx] > 0).float()
            donor_count = donor_labels.sum().item()
            
            intersection = (anchor_labels * donor_labels).sum().item()
            union = anchor_count + donor_count - intersection
            
            if is_dc_branch:
                unique_in_donor = donor_count - intersection
                shared_bonus = min(intersection, 2) / 3.0
                if intersection > 2:
                    shared_bonus = max(0, 2.0 - (intersection - 2) * 0.3) / 3.0
                relevance = unique_in_donor * (1.0 + shared_bonus)
            else:
                relevance = intersection / union if union > 0 else 0.0
            
            relevance_scores.append(relevance)
        
        return torch.tensor(relevance_scores, device=targets.device)

    def _reorder_donors_by_saliency_relevance(self, inputs, anchor_idx, donor_indices, relevance_scores, g_row, g_col):
        """
        Reorder donors: high-saliency grid positions receive high-relevance donors.
        """
        n_donors = len(donor_indices)
        if n_donors <= 1:
            return donor_indices
        
        g = g_row * g_col
        anchor_img = inputs[anchor_idx:anchor_idx+1]
        _, _, H, W = anchor_img.shape
        grid_h, grid_w = H // g_row, W // g_col
        
        sal_map, _, _, _ = self._compute_saliency(anchor_img)
        sal_map = sal_map[0]
        
        grid_saliencies = []
        for row in range(g_row):
            for col in range(g_col):
                region = sal_map[row*grid_h:(row+1)*grid_h, col*grid_w:(col+1)*grid_w]
                grid_saliencies.append(region.mean().item())
        
        donor_positions = list(range(1, g))
        donor_grid_saliencies = [grid_saliencies[p] for p in donor_positions]
        
        pos_sal_pairs = list(zip(donor_positions, donor_grid_saliencies))
        pos_sal_pairs.sort(key=lambda x: x[1], reverse=True)
        sorted_positions = [p for p, _ in pos_sal_pairs]
        
        rel_scores = relevance_scores.cpu().numpy()
        donor_rel_pairs = list(zip(donor_indices, rel_scores))
        donor_rel_pairs.sort(key=lambda x: x[1], reverse=True)
        sorted_donors = [d for d, _ in donor_rel_pairs]
        
        position_to_donor = {}
        for i, pos in enumerate(sorted_positions):
            if i < len(sorted_donors):
                position_to_donor[pos] = sorted_donors[i]
        
        reordered_donors = []
        for pos in donor_positions:
            reordered_donors.append(position_to_donor.get(pos, donor_indices[pos-1]))
        
        return reordered_donors

    def _get_tiered_dc_reduction(self, anchor_classes):
        """
        V2: Get tiered DC reduction factor.
        极稀有(P10)->0.3, 稀有(P25)->0.6, 常见->1.0
        """
        if not self.rare_class_aware:
            return 1.0
        
        has_very_rare = False
        has_rare = False
        
        for cls_idx in anchor_classes:
            cls_id = cls_idx.item() if hasattr(cls_idx, "item") else cls_idx
            if self.use_percentile_threshold and (self.rare_classes_p10 or self.rare_classes_p25):
                if cls_id in self.rare_classes_p10:
                    has_very_rare = True
                    break
                elif cls_id in self.rare_classes_p25:
                    has_rare = True
            else:
                if hasattr(self, "class_freq") and cls_id < len(self.class_freq):
                    if self.class_freq[cls_id] < self.rare_class_threshold:
                        has_rare = True
        
        if has_very_rare:
            return self.dc_reduction_very_rare
        elif has_rare:
            return self.dc_reduction_rare
        return 1.0
    
    def set_dataset(self, dataset_name):
        """Set dataset and load corresponding cooccur file."""
        import os
        base_dir = os.path.dirname(__file__)
        if "coco" in dataset_name.lower() or "ms-coco" in dataset_name.lower():
            self.group_meta_path = os.path.join(base_dir, "cooccur_coco.json")
            print(f"[SpliceMix V2] Using COCO meta")
        else:
            self.group_meta_path = os.path.join(base_dir, "cooccur_voc2007.json")
            print(f"[SpliceMix V2] Using VOC meta")
        if os.path.exists(self.group_meta_path):
            self.group_meta = self._load_group_meta(self.group_meta_path)
    
    def Smix(self, inputs, targets, ):
        if np.random.rand(1) > self.mix_prob:
            return inputs, targets, {}
        if self.Default:
            coin = random.random()
            coin_dp = random.random()
            self.n_grids = [inputs.shape[0]//4, ]
            ## defaut_max_2  84.78 SplicMix-CL, 84.11 SpliceMix in coco
            if coin > self.config_default['1x2']:
                n_drop = 1 if coin_dp < self.config_default['drop_rate'] else 0
                self.grids = [f'1x2-{n_drop}',]
            elif coin > self.config_default['2x2']:
                n_drop = random.sample(range(1, 4), 1)[0] if coin_dp < self.config_default['drop_rate'] else 0
                self.grids = [f'2x2-{n_drop}', ]
            elif coin > self.config_default['2x3']:
                n_drop = random.sample(range(1, 6), 1)[0] if coin_dp < self.config_default['drop_rate'] else 0
                self.grids = [f'2x3-{n_drop}', ]

        # --- saliency-guided donor selection (non-gradient, input-level) ---
        use_sal = False
        sal_scores = None
        if self.use_saliency:
            sal_map, sal_scores, sal_entropy, sal_var = self._compute_saliency(inputs)
            # adapt top-p by label density (more positives -> higher top-p)
            label_density = float(targets.mean()) if targets.numel() > 0 else 0.0
            top_p = self.saliency_top_p_min + (self.saliency_top_p_max - self.saliency_top_p_min) * min(label_density / 0.5, 1.0)
            saliency_top_p_curr = max(self.saliency_top_p_min, min(self.saliency_top_p_max, top_p))
            # fallback when saliency is nearly flat
            if sal_var.mean() > self.saliency_var_threshold:
                use_sal = True
        else:
            saliency_top_p_curr = self.saliency_top_p
        # group-aware mode: decide conservative vs diverse based on label group coverage and strong co-occurrence hits
        group_mode = None
        rel_frac, dis_frac, strong_frac = 0.0, 0.0, 0.0
        strong_adj = None
        if self.group_mix_enable and getattr(self, 'group_meta', {}):
            related = self.group_meta.get('related', [])
            discriminative = self.group_meta.get('discriminative', [])
            strong_pairs = self.group_meta.get('strong_pairs', [])
            strong_per_class = self.group_meta.get('strong_per_class', {})
            label_pos = (targets > 0).float()
            rel_mask = label_pos[:, related].max(dim=1).values if len(related) > 0 else torch.zeros_like(targets[:, 0])
            dis_mask = label_pos[:, discriminative].max(dim=1).values if len(discriminative) > 0 else torch.zeros_like(targets[:, 0])
            rel_frac = rel_mask.mean().item()
            dis_frac = dis_mask.mean().item()

            # build strong adjacency from meta for fast batch hit counting
            C = targets.shape[1]
            strong_adj = torch.zeros((C, C), device=targets.device, dtype=torch.bool)
            if strong_per_class:
                for k, v in strong_per_class.items():
                    i = int(k)
                    for j in v:
                        if 0 <= i < C and 0 <= j < C:
                            strong_adj[i, j] = True
            elif strong_pairs:
                for i, j in strong_pairs:
                    if 0 <= i < C and 0 <= j < C:
                        strong_adj[i, j] = True
            if strong_adj.any():
                mask = label_pos.bool().float()
                pair_hit = (mask @ strong_adj.float()) > 0  # classes that have a strong partner present in the same sample
                strong_frac = pair_hit.any(dim=1).float().mean().item()

            rel_signal = max(rel_frac, min(1.0, strong_frac * 1.5))
            if rel_signal > dis_frac:
                group_mode = 'related'
            elif dis_frac > rel_signal:
                group_mode = 'diverse'
        
        # ===== CDM: Curriculum Diversity Mixing =====
        if self.cdm_enable:
            if self.current_epoch < self.cdm_warmup_epochs:
                # 早期强制related模式
                group_mode = "related"
            else:
                # 计算 epoch ratio (从 warmup 结束后开始)
                effective_epoch = self.current_epoch - self.cdm_warmup_epochs
                remaining_epochs = max(1, self.total_epochs - self.cdm_warmup_epochs)
                epoch_ratio = min(1.0, effective_epoch / remaining_epochs)
                
                # Cosine 调度：更平滑的增长
                if self.cdm_cosine_schedule:
                    import math
                    # cosine: 从0到1的平滑增长
                    cosine_ratio = 0.5 * (1 - math.cos(math.pi * epoch_ratio))
                    cdm_diverse_prob = self.cdm_diverse_prob_min + (self.cdm_diverse_prob_max - self.cdm_diverse_prob_min) * cosine_ratio
                else:
                    cdm_diverse_prob = self.cdm_diverse_prob_min + (self.cdm_diverse_prob_max - self.cdm_diverse_prob_min) * epoch_ratio
                
                # 稀有类保护：如果 batch 中有稀有类，降低 diverse 概率
                if self.cdm_rare_protection and hasattr(self, 'very_rare_classes'):
                    batch_has_rare = targets[:, list(self.very_rare_classes) + list(self.rare_classes)].sum() > 0
                    if batch_has_rare:
                        cdm_diverse_prob = cdm_diverse_prob * self.cdm_rare_diverse_scale
                
                if random.random() < cdm_diverse_prob:
                    group_mode = "diverse"
        # choose grids and top-p scaling
        grids_to_use = self.grids
        n_grids_to_use = self.n_grids if len(self.n_grids) == len(grids_to_use) else [self.n_grids[0]] * len(grids_to_use)
        top_p_scale = 1.0
        if group_mode == 'related':
            grids_to_use = self.group_grids_related
            n_grids_to_use = [self.n_grids[0]] * len(grids_to_use)
            top_p_scale = self.group_top_p_scale_related
        elif group_mode == 'diverse':
            grids_to_use = self.group_grids_diverse
            n_grids_to_use = [self.n_grids[0]] * len(grids_to_use)
            top_p_scale = self.group_top_p_scale_diverse
        saliency_top_p_curr = saliency_top_p_curr * top_p_scale
        if self.group_log_prob > 0 and random.random() < self.group_log_prob:
            print(f'[SpliceMix][group] mode={group_mode}, rel_frac={rel_frac:.3f}, dis_frac={dis_frac:.3f}, strong_frac={strong_frac:.3f}, top_p={saliency_top_p_curr:.3f}')

        bs = inputs.shape[0]
        mix_ind = torch.zeros((bs), device=inputs.device)
        mix_dict = {'rand_inds': [], 'rows': [], 'cols': [], 'n_drops': [], 'drop_inds': []}
        
        # Build co-occurrence adjacency matrix for label-compatible donor selection
        C = targets.shape[1]
        cooccur_adj = self._build_cooccur_adj(C, targets.device)
        
        # IMPORTANT: Save original inputs/targets for donor selection (before cat expansion)
        inputs_orig = inputs
        targets_orig = targets.clone()
        
        # Note: Each mixed group selects donors independently (intra-group unique only)
        
        for g, ng in zip(grids_to_use, n_grids_to_use):
            g_row, g_col = [int(t) if '-' not in t else t.split('-') for t in g.split('x')]
            (g_col, n_drop) = [int(t) for t in g_col] if type(g_col) is list else (g_col, 0)
            g = g_row * g_col
            if ng == 0:
                if len(self.grids) == 1:
                    ng = bs // g
                else:
                    raise AssertionError('argument error, cannot execute c-mix')
            
            # CO/DC Dual-Branch Label-Aware Donor Selection
            # CO (Correlative): select label-compatible donors to reinforce label co-occurrence
            # DC (Discriminative): select label-incompatible donors to enhance feature separability
            rand_ind_g_list = []
            branch_counts = {'co': 0, 'dc': 0}
            
            for _ in range(ng):
                # Each group selects independently (only avoid intra-group duplicates)
                group_used = set()
                
                # Select anchor: prefer high saliency if available, else random
                available = list(range(bs))
                
                if use_sal:
                    # Pick anchor from top saliency among available
                    score = sal_scores + self.saliency_entropy_weight * sal_entropy
                    temp = torch.clamp((sal_var / (self.saliency_var_threshold * 10 + self.saliency_eps)).sqrt(),
                                       min=self.saliency_temp_min, max=self.saliency_temp_max)
                    score = score / temp
                    avail_scores = [(i, score[i].item()) for i in available]
                    avail_scores.sort(key=lambda x: x[1], reverse=True)
                    top_k = max(int(len(avail_scores) * saliency_top_p_curr), 1)
                    anchor_idx = random.choice([x[0] for x in avail_scores[:top_k]])
                else:
                    anchor_idx = random.choice(available)
                
                group_used.add(anchor_idx)
                
                # ===== CO/DC Dual-Branch Decision =====
                # Compute anchor's label density for adaptive DC probability
                anchor_label_count = (targets_orig[anchor_idx] > 0).sum().item()
                anchor_density = anchor_label_count / targets_orig.shape[1]  # normalize by num_classes
                
                # Curriculum Learning + Density-Adaptive DC probability
                # Key insight: Early training needs CO (learn patterns), late training needs DC (prevent overfitting)
                if self.dual_branch_enable:
                    # Base DC prob from curriculum (linear increase over epochs)
                    if self.dc_curriculum_enable:
                        epoch_ratio = min(1.0, self.current_epoch / max(1, self.total_epochs))
                        curriculum_dc = self.dc_curriculum_min + (self.dc_curriculum_max - self.dc_curriculum_min) * epoch_ratio
                    else:
                        curriculum_dc = self.dc_base_prob
                    
                    # Optional: also consider label density (additive bonus)
                    if self.dc_adaptive:
                        density_bonus = self.dc_density_scale * anchor_density * 0.3  # scaled down
                        dc_prob = min(0.8, curriculum_dc + density_bonus)
                    else:
                        dc_prob = curriculum_dc
                    
                    # V2: Tiered Rare-Class Aware DC Reduction
                    # 极稀有(P10) -> 0.3x, 稀有(P25) -> 0.6x
                    if self.rare_class_aware and hasattr(self, "class_freq") and len(self.class_freq) >= targets_orig.shape[1]:
                        anchor_classes = (targets_orig[anchor_idx] > 0).nonzero(as_tuple=True)[0]
                        dc_reduction = self._get_tiered_dc_reduction(anchor_classes)
                        dc_prob = dc_prob * dc_reduction
                    use_dc_branch = random.random() < dc_prob
                else:
                    use_dc_branch = False
                
                # Select donors based on branch choice
                if use_dc_branch:
                    # DC Branch: select discriminative (label-incompatible) donors
                    donors = self._select_discriminative_donors(
                        targets_orig, anchor_idx, g - 1, cooccur_adj, used_set=group_used.copy()
                    )
                    branch_counts['dc'] += 1
                    is_dc = True
                else:
                    # CO Branch: select compatible (label-correlated) donors
                    donors = self._select_compatible_donors(
                        targets_orig, anchor_idx, g - 1, cooccur_adj, used_set=group_used.copy()
                    )
                    branch_counts['co'] += 1
                    is_dc = False
                
                # V3: Saliency-Relevance Matching
                # Reorder donors so high-saliency grid positions get high-relevance donors
                if len(donors) > 1:
                    relevance = self._compute_relevance(targets_orig, anchor_idx, donors, is_dc_branch=is_dc)
                    donors = self._reorder_donors_by_saliency_relevance(
                        inputs_orig, anchor_idx, donors, relevance, g_row, g_col
                    )
                
                for d in donors:
                    group_used.add(d)
                
                # Combine anchor + donors
                group_indices = [anchor_idx] + donors
                rand_ind_g_list.extend(group_indices)
            
            rand_ind_g = np.asarray(rand_ind_g_list)
            
            # Log CO/DC dual-branch stats
            if hasattr(self, 'compat_log_prob') and self.compat_log_prob > 0 and random.random() < self.compat_log_prob:
                stats_co = getattr(self, '_last_compat_stats', {})
                stats_dc = getattr(self, '_last_dc_stats', {})
                group_targets = targets_orig[rand_ind_g]
                shared_any = ((group_targets > 0).float().sum(dim=0) > 1).sum().item()
                print(f'[SpliceMix][dual-branch] grid={g_row}x{g_col}, ng={ng}, '
                      f'CO={branch_counts["co"]}, DC={branch_counts["dc"]}, '
                      f'dc_prob={dc_prob:.3f}, shared_classes={shared_any}')
            
            if g_row != g_col and self.use_asym:  # for asymmetric grids
                if np.random.randn() < 0: g_row, g_col = g_col, g_row
            # CAM-Guided Mix: 使用 CAM 引导的混合策略
            if self.cam_guided_enable and use_sal:
                sal_map_group = sal_map[rand_ind_g] if sal_map is not None else None
                inputs_mix_g, targets_mix_g, cam_info = self.mix_fn_cam_guided(
                    inputs[rand_ind_g], targets[rand_ind_g],
                    g_row=g_row, g_col=g_col, n_grid=ng,
                    sal_map=sal_map_group, n_drop=n_drop
                )
                drop_ind = torch.zeros((len(rand_ind_g)), device=inputs.device)
            else:
                inputs_mix_g, targets_mix_g, drop_ind = self.mix_fn(
                    inputs[rand_ind_g], targets[rand_ind_g],
                    g_row=g_row, g_col=g_col, n_grid=ng, n_drop=n_drop,
                    sal_scores=None if sal_scores is None else sal_scores[rand_ind_g].view(ng, g)
                )

            # optional visualization of mixed outputs
            self._maybe_save_viz(inputs_mix_g, targets_mix_g, tag=f'mix_{g_row}x{g_col}_ng{ng}')

            inputs = torch.cat([inputs, inputs_mix_g], dim=0)
            targets = torch.cat([targets, targets_mix_g], dim=0)
            mix_dict['rand_inds'].append(rand_ind_g)
            mix_dict['rows'].append(g_row)
            mix_dict['cols'].append(g_col)
            mix_dict['n_drops'].append(n_drop)
            mix_dict['drop_inds'].append(drop_ind)  # the index in a mixed image, e.g., for a 2x2 grid, len(drop_ind)=4 || rand_ind_g[bool(drop_ind)] back to the index of dropped regular images
            mix_ind = torch.cat([mix_ind, torch.ones((ng), device=mix_ind.device)], dim=0)
        flag = {'mix_ind': mix_ind, 'mix_dict': mix_dict, }
        return inputs, targets, flag

    def _maybe_save_viz(self, inputs_mix, targets_mix, tag):
        if not self.viz_enable or self.viz_count >= self.viz_max:
            return
        if random.random() > self.viz_prob:
            return
        os.makedirs(self.viz_dir, exist_ok=True)
        img = inputs_mix[0:1].detach().cpu()
        img = img - img.min()
        img = img / (img.max() + 1e-6)
        fname = f"{tag}_{self.viz_count:04d}_{int(time.time())}.png"
        try:
            torchvision.utils.save_image(img, os.path.join(self.viz_dir, fname))
            self.viz_count += 1
        except Exception as e:
            print(f"[SpliceMix][viz] save failed: {e}")


    def mix_fn(self, inputs, targets, g_row, g_col, n_grid, n_drop=0, sal_scores=None):
        bs, c, h, w = inputs.shape
        g = g_row * g_col
        drop_ind = torch.zeros((bs), device=inputs.device)
        if n_drop > 0:
            if sal_scores is not None:
                # drop lowest-saliency donors inside each mixed sample
                sal_rank = torch.argsort(sal_scores, dim=1)  # ng, g
                drop_list = []
                for i in range(n_grid):
                    drop_list.append(sal_rank[i, :n_drop])
                drop_rand_ind = torch.stack(drop_list, dim=0).reshape(-1).cpu().numpy()
            else:
                drop_rand_ind = np.asarray([random.sample(range(i*g, (i+1)*g), n_drop) for i in range(n_grid)]).reshape(-1)
            drop_ind[drop_rand_ind] = 1
            inputs = inputs * (1 - drop_ind[:, None, None, None])
        inputs = F.interpolate(inputs, (h // g_row, w // g_col), mode='bilinear', align_corners=True)  # g*ng, C, h', w'
        inputs_mix = torchvision.utils.make_grid(inputs, nrow=g_col, padding=0)  # C, ng*h, w
        inputs_mix = inputs_mix.split(h//g_row * g_row, dim=1)  # tuple: ng, (C, h, w)
        inputs_mix = torch.stack(inputs_mix, dim=0)
        
        if (inputs_mix.shape[-2], inputs_mix.shape[-1]) != (h, w):
            inputs_mix = F.interpolate(inputs_mix, (h, w), mode='bilinear', align_corners=True)

        if n_drop > 0:
            targets = targets * (1 - drop_ind[:, None])
        targets_mix = targets.view(n_grid, g, -1).sum(1)  # ng, nc
        targets_mix[targets_mix > 0] = 1

        return inputs_mix, targets_mix, drop_ind




    def mix_fn_cam_guided(self, inputs, targets, g_row, g_col, n_grid, sal_map=None, n_drop=0):
        """
        CAM 引导的混合函数
        
        与原始 mix_fn 的区别：
        1. 不是简单拼接，而是根据 CAM 决定每个 grid 的操作
        2. 支持可控遮挡（训练抗遮挡能力）
        3. 支持语义对齐的部件替换
        """
        bs, c, h, w = inputs.shape
        g = g_row * g_col
        
        # 如果没有 saliency map 或 CAM 引导未启用，回退到原始 mix_fn
        if sal_map is None or not self.cam_guided_enable:
            inputs_mix, targets_mix, drop_ind = self.mix_fn(
                inputs, targets, g_row, g_col, n_grid, n_drop=n_drop
            )
            return inputs_mix, targets_mix, {"fallback": True}
        
        # 初始化 CAM mixer
        if self.cam_mixer is None:
            self.cam_mixer = CAMGuidedMixerV2()
        
        # 存储混合结果
        inputs_mix_list = []
        targets_mix_list = []
        mix_stats = {"keep_anchor": 0, "use_donor": 0, "occlude": 0, "cam_guided": 0, "fallback": 0, "total_grids": 0}
        
        for i in range(n_grid):
            start_idx = i * g
            end_idx = (i + 1) * g
            group_inputs = inputs[start_idx:end_idx]
            group_sal = sal_map[start_idx:end_idx]
            group_targets = targets[start_idx:end_idx]
            
            # V2: 使用 targets_group 而不是 sal_map
            mixed, mix_info = self.cam_mixer.apply_cam_guided_mix(
                group_inputs, group_targets, g_row, g_col,
                self.current_epoch, self.total_epochs, cam=None
            )
            
            inputs_mix_list.append(mixed)
            target_mix = group_targets.sum(dim=0)
            target_mix[target_mix > 0] = 1
            targets_mix_list.append(target_mix)
            
            for k, v in mix_info["stats"].items():
                mix_stats[k] += v
        
        inputs_mix = torch.stack(inputs_mix_list, dim=0)
        targets_mix = torch.stack(targets_mix_list, dim=0)
        
        if hasattr(self, "cam_mixer") and random.random() < 0.02:
            total = sum(mix_stats.values())
            print(f"[CAM-Mix] epoch={self.current_epoch}, anchor={mix_stats.get('keep_anchor', 0)}, donor={mix_stats.get('use_donor', 0)}, occlude={mix_stats.get('occlude', 0)}, cam_guided={mix_stats.get('cam_guided', 0)}")
        
        return inputs_mix, targets_mix, {"stats": mix_stats, "fallback": False}

    def _compute_saliency(self, inputs):
        # Multi-scale, non-gradient saliency using feature norm + Sobel edges.
        B, C, H, W = inputs.shape
        sal_agg = None
        for scale in self.saliency_scales:
            if scale != 1.0:
                scaled = F.interpolate(inputs, scale_factor=scale, mode='bilinear', align_corners=True)
            else:
                scaled = inputs

            r, g, b = scaled[:, 0:1], scaled[:, 1:2], scaled[:, 2:3]
            gray = 0.299 * r + 0.587 * g + 0.114 * b

            sobel_kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=inputs.dtype, device=inputs.device).view(1, 1, 3, 3)
            sobel_kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=inputs.dtype, device=inputs.device).view(1, 1, 3, 3)
            edge_x = F.conv2d(gray, sobel_kernel_x, padding=1)
            edge_y = F.conv2d(gray, sobel_kernel_y, padding=1)
            edge_mag = torch.sqrt(edge_x ** 2 + edge_y ** 2 + self.saliency_eps)

            feat_norm = torch.sqrt((scaled ** 2).sum(dim=1, keepdim=True) + self.saliency_eps)
            sal = self.saliency_weights['norm'] * feat_norm + self.saliency_weights['edge'] * edge_mag

            k = self.saliency_smooth_kernel
            if k > 1:
                sal = F.avg_pool2d(sal, kernel_size=k, stride=1, padding=k // 2)

            if sal.shape[-2:] != (H, W):
                sal = F.interpolate(sal, size=(H, W), mode='bilinear', align_corners=True)

            sal_agg = sal if sal_agg is None else sal_agg + sal

        sal = sal_agg / len(self.saliency_scales)
        sal = sal - sal.amin(dim=[1, 2, 3], keepdim=True)
        sal = sal / (sal.amax(dim=[1, 2, 3], keepdim=True) + self.saliency_eps)

        sal_entropy = -(sal * (sal + self.saliency_eps).log()).mean(dim=[1, 2, 3])
        # spatial variance per image (reduce H,W)
        sal_var = sal.var(dim=[2, 3]).mean(dim=1)
        sal_scores = sal.mean(dim=[1, 2, 3])  # B
        sal = sal.squeeze(1)  # B,H,W
        return sal, sal_scores, sal_entropy, sal_var
    def Smix_minimalism(self, X, Y):
        g_row, g_col = 2, 2
        B, C, H, W = X.shape
        ng = B // (g_row * g_col) * (g_row * g_col)
        Omega = random.sample(range(B), B//ng)
        X_ds = F.interpolate(X[Omega], (H // g_row, W // g_col), mode='bilinear', align_corners=True)  # g*ng, C, h', w'
        X_ = torchvision.utils.make_grid(X_ds, nrow=g_col, padding=0)  # C, ng*h, w
        X_ = X_.split(H, dim=1)  # tuple: ng, (C, h, w)
        X_ = torch.stack(X_, dim=0)  # ng, C, H, W
        Y_ = Y[Omega].view(ng, g_row * g_col, -1).sum(1)
        Y_[Y_ > 0] = 1

        X_hat = torch.cat((X, X_), dim=0)
        Y_hat = torch.cat((Y, Y_), dim=0)
        return X_hat, Y_hat, {}

    def checkMode(self, mode):
        if '--' in mode:  # like Splice--Mini=True
            str_list = mode.split('--')
            # mode = str_list[0]
            for s in str_list[1:]:
                exec(f"self.{s}")
        # return mode


def get_imgs(dir, bs=16, num_classes=10):
    transf = transforms.Compose([
        transforms.RandomResizedCrop(448, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    imgs = os.listdir(img_dir)
    inputs = torch.tensor([])

    for i in range(bs):
        img_path = os.path.join(dir, imgs[i])
        input = Image.open(img_path).convert('RGB')
        inputs = torch.cat((inputs, transf(input).unsqueeze(0)), 0)

    tgts = torch.rand(bs, num_classes)
    tgts[tgts>.7] = 1
    tgts[tgts<1] = 0
    return inputs, tgts

def plt_imgs(imgs, rows=2, tgts=None):
    plt.figure()
    for i in range(imgs.shape[0]):
        plt.subplot(rows, int(np.ceil(imgs.shape[0] / rows)), i + 1)
        plt.imshow(imgs[i].permute(1, 2, 0))
        plt.title(str(i))
        # plt.axis('off')
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
        if tgts is not None:
            plt.xlabel(tgts[i])
    plt.show(block=False)

if __name__ == '__main__':
    import os
    import PIL.Image as Image
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    import numpy as np
    img_dir = 'E:\PhD\Data_set\ImageSet\VOC2007\VOCdevkit\VOC2007\JPEGImages'

    bs = 8
    imgs, ptgts = get_imgs(dir=img_dir, bs=bs)
    # imgs, ptgts = imgs.cuda(), ptgts.cuda()
    print(ptgts, ptgts.sum(-1))
    mixer = SpliceMix(mode='SpliceMix', grids=['1x2', '2x3-2'], n_grids=[1, 2]).mixer
    imgs_mix, tgts_mix, flag = mixer(imgs, ptgts)
    print(flag)
    print(tgts_mix[-5:])
    plt_imgs(imgs_mix.cpu(), tgts=tgts_mix.cpu().numpy())











# ============================================================================
# Module 2: CAM-Guided Part-Level Mix (CAM 引导的部件级混合)
# ============================================================================
# 动机：
# 1. 传统随机裁拼容易打断语义结构，把无关背景贴到目标上
# 2. 用 CAM 定位类相关区域，在关键部件间做可控混合/遮挡
# 3. 让模型学到部件级的抗遮挡能力和组合泛化能力
# ============================================================================

class CAMGuidedMixerV2:
    """CAM-Guided Part-Level Mix Module V2"""
    
    def __init__(self):
        self.cam_guided_enable = False
        self.classifier_weights = None
        self.num_classes = 20
        self.feat_dim = 2048
        self.cam_smooth_kernel = 5
        self.cam_normalize = True
        self.conf_threshold = 0.4
        self.min_confident_classes = 1
        self.class_consistency_enable = True
        self.class_overlap_min = 1
        self.use_ema_thresholds = True
        self.ema_momentum = 0.95
        self.ema_high_threshold = 0.7
        self.ema_low_threshold = 0.3
        self.quantile_high = 0.80
        self.quantile_low = 0.20
        self.occlusion_enable = True
        self.occlusion_prob_max = 0.15
        self.occlusion_prob_min = 0.0
        self.occlusion_warmup = 20
        self.occlusion_value = 0.0
        self.use_cosine_schedule = True
        self.cam_strength_min = 0.3
        self.cam_strength_max = 1.0
        self.safety_fallback_enable = True
        self.cam_variance_threshold = 0.02
        self.max_fallback_ratio = 0.8
        self.log_prob = 0.02
        self.log_enabled = True
        self.stats_window = []
        self.stats_window_size = 100

    def set_classifier_weights(self, model):
        try:
            if hasattr(model, 'module'): model = model.module
            if hasattr(model, 'cls') and hasattr(model.cls, 'weight'):
                self.classifier_weights = model.cls.weight.data.clone()
                self.num_classes = self.classifier_weights.shape[0]
                self.feat_dim = self.classifier_weights.shape[1]
                print(f"[CAM-V2] Loaded classifier weights: {self.classifier_weights.shape}")
                return True
        except Exception as e:
            print(f"[CAM-V2] Failed to load classifier weights: {e}")
        return False

    def compute_lightweight_cam(self, inputs):
        B, C, H, W = inputs.shape
        device = inputs.device
        intensity = torch.sqrt((inputs ** 2).sum(dim=1) + 1e-6)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=inputs.dtype, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=inputs.dtype, device=device).view(1, 1, 3, 3)
        gray = inputs.mean(dim=1, keepdim=True)
        edge_x = F.conv2d(gray, sobel_x, padding=1)
        edge_y = F.conv2d(gray, sobel_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6).squeeze(1)
        cam = 0.6 * intensity + 0.4 * edge
        if self.cam_smooth_kernel > 1:
            k = self.cam_smooth_kernel
            cam = F.avg_pool2d(cam.unsqueeze(1), kernel_size=k, stride=1, padding=k//2).squeeze(1)
        cam_min = cam.view(B, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
        cam_max = cam.view(B, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
        return cam

    def compute_grid_cam(self, cam, g_row, g_col):
        B, H, W = cam.shape
        grid_cam = F.adaptive_avg_pool2d(cam.unsqueeze(1), (g_row, g_col)).squeeze(1)
        return grid_cam

    def check_class_consistency(self, anchor_targets, donor_targets):
        if not self.class_consistency_enable:
            return list(range(donor_targets.shape[0]))
        valid_donors = []
        for i in range(donor_targets.shape[0]):
            overlap = ((anchor_targets > 0) & (donor_targets[i] > 0)).sum().item()
            if overlap >= self.class_overlap_min:
                valid_donors.append(i)
        return valid_donors

    def update_ema_thresholds(self, grid_cam):
        if not self.use_ema_thresholds: return
        flat_cam = grid_cam.flatten()
        current_high = flat_cam.quantile(self.quantile_high).item()
        current_low = flat_cam.quantile(self.quantile_low).item()
        self.ema_high_threshold = self.ema_momentum * self.ema_high_threshold + (1 - self.ema_momentum) * current_high
        self.ema_low_threshold = self.ema_momentum * self.ema_low_threshold + (1 - self.ema_momentum) * current_low

    def get_schedule_values(self, current_epoch, total_epochs):
        epoch_ratio = min(1.0, current_epoch / max(1, total_epochs))
        if self.use_cosine_schedule:
            cos_ratio = 0.5 * (1 - np.cos(np.pi * epoch_ratio))
            cam_strength = self.cam_strength_min + (self.cam_strength_max - self.cam_strength_min) * cos_ratio
            if current_epoch < self.occlusion_warmup:
                occlusion_prob = 0.0
            else:
                occ_ratio = min(1.0, (current_epoch - self.occlusion_warmup) / max(1, total_epochs - self.occlusion_warmup))
                occ_cos = 0.5 * (1 - np.cos(np.pi * occ_ratio))
                occlusion_prob = self.occlusion_prob_min + (self.occlusion_prob_max - self.occlusion_prob_min) * occ_cos
        else:
            cam_strength = self.cam_strength_min + (self.cam_strength_max - self.cam_strength_min) * epoch_ratio
            occlusion_prob = 0.0 if current_epoch < self.occlusion_warmup else self.occlusion_prob_min + (self.occlusion_prob_max - self.occlusion_prob_min) * min(1.0, (current_epoch - self.occlusion_warmup) / max(1, total_epochs - self.occlusion_warmup))
        return cam_strength, occlusion_prob

    def check_cam_quality(self, grid_cam):
        if not self.safety_fallback_enable: return True, "safety_disabled"
        flat_cam = grid_cam.flatten()
        cam_var = flat_cam.var().item()
        if cam_var < self.cam_variance_threshold: return False, f"low_variance({cam_var:.4f})"
        cam_range = flat_cam.max().item() - flat_cam.min().item()
        if cam_range < 0.1: return False, f"narrow_range({cam_range:.4f})"
        return True, "valid"

    def apply_cam_guided_mix(self, inputs_group, targets_group, g_row, g_col, current_epoch, total_epochs, cam=None, predictions=None):
        n_images, C, H, W = inputs_group.shape
        g = g_row * g_col
        device = inputs_group.device
        h_step, w_step = H // g_row, W // g_col
        cam_strength, occlusion_prob = self.get_schedule_values(current_epoch, total_epochs)
        stats = {"keep_anchor": 0, "use_donor": 0, "occlude": 0, "cam_guided": 0, "fallback": 0, "total_grids": g}
        if cam is None: cam = self.compute_lightweight_cam(inputs_group)
        grid_cam = self.compute_grid_cam(cam, g_row, g_col)
        anchor_grid_cam = grid_cam[0]
        cam_valid, cam_reason = self.check_cam_quality(anchor_grid_cam)
        if not cam_valid:
            stats["fallback"] = g
            mixed = self._fallback_splice(inputs_group, g_row, g_col)
            return mixed, {"stats": stats, "cam_strength": cam_strength, "occlusion_prob": occlusion_prob, "fallback_reason": cam_reason}
        self.update_ema_thresholds(anchor_grid_cam)
        high_thresh, low_thresh = self.ema_high_threshold, self.ema_low_threshold
        anchor_targets = targets_group[0]
        donor_targets = targets_group[1:] if n_images > 1 else None
        valid_donors = self.check_class_consistency(anchor_targets, donor_targets) if donor_targets is not None else []
        mixed = torch.zeros((C, H, W), device=device, dtype=inputs_group.dtype)
        for i in range(g_row):
            for j in range(g_col):
                grid_idx = i * g_col + j
                h_start, h_end = i * h_step, (i + 1) * h_step
                w_start, w_end = j * w_step, (j + 1) * w_step
                anchor_cam_val = anchor_grid_cam[i, j].item()
                effective_high = high_thresh * cam_strength + (1 - cam_strength) * 0.5
                effective_low = low_thresh * cam_strength + (1 - cam_strength) * 0.5
                if anchor_cam_val > effective_high:
                    if self.occlusion_enable and random.random() < occlusion_prob:
                        mixed[:, h_start:h_end, w_start:w_end] = self.occlusion_value
                        stats["occlude"] += 1
                    else:
                        mixed[:, h_start:h_end, w_start:w_end] = inputs_group[0, :, h_start:h_end, w_start:w_end]
                        stats["keep_anchor"] += 1
                elif anchor_cam_val < effective_low and len(valid_donors) > 0:
                    donor_cams = [grid_cam[d+1, i, j].item() for d in valid_donors]
                    best_donor_idx = valid_donors[np.argmax(donor_cams)]
                    if max(donor_cams) > anchor_cam_val + 0.1:
                        mixed[:, h_start:h_end, w_start:w_end] = inputs_group[best_donor_idx + 1, :, h_start:h_end, w_start:w_end]
                        stats["use_donor"] += 1
                        stats["cam_guided"] += 1
                    else:
                        src_idx = grid_idx % n_images
                        mixed[:, h_start:h_end, w_start:w_end] = inputs_group[src_idx, :, h_start:h_end, w_start:w_end]
                        stats["keep_anchor" if src_idx == 0 else "use_donor"] += 1
                else:
                    src_idx = grid_idx % n_images
                    mixed[:, h_start:h_end, w_start:w_end] = inputs_group[src_idx, :, h_start:h_end, w_start:w_end]
                    stats["keep_anchor" if src_idx == 0 else "use_donor"] += 1
        mix_info = {"stats": stats, "cam_strength": cam_strength, "occlusion_prob": occlusion_prob, "ema_high": self.ema_high_threshold, "ema_low": self.ema_low_threshold, "fallback_reason": None}
        if self.log_enabled and random.random() < self.log_prob:
            print(f"[CAM-V2] epoch={current_epoch}, strength={cam_strength:.2f}, occ_prob={occlusion_prob:.3f}, anchor={stats['keep_anchor']}, donor={stats['use_donor']}, occlude={stats['occlude']}, cam_guided={stats['cam_guided']}, ema_h={self.ema_high_threshold:.3f}, ema_l={self.ema_low_threshold:.3f}")
        return mixed, mix_info

    def _fallback_splice(self, inputs_group, g_row, g_col):
        n_images, C, H, W = inputs_group.shape
        h_step, w_step = H // g_row, W // g_col
        mixed = torch.zeros((C, H, W), device=inputs_group.device, dtype=inputs_group.dtype)
        for i in range(g_row):
            for j in range(g_col):
                grid_idx = i * g_col + j
                h_start, h_end = i * h_step, (i + 1) * h_step
                w_start, w_end = j * w_step, (j + 1) * w_step
                src_idx = grid_idx % n_images
                mixed[:, h_start:h_end, w_start:w_end] = inputs_group[src_idx, :, h_start:h_end, w_start:w_end]
        return mixed


CAMGuidedMixer = CAMGuidedMixerV2
_cam_guided_mixer = CAMGuidedMixerV2()

