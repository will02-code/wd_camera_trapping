import torch
import numpy as np

class CustomConfusionMatrix:
    """
    Custom confusion matrix that stores all detections (including low conf)
    and allows re-computing the confusion matrix at different confidence thresholds.
    """

    def __init__(self, nc, iou_thres=0.45, task="detect"):
        """
        Args:
            nc (int): Number of classes (not counting 'background').
            iou_thres (float): IoU threshold to decide TPs vs. FPs.
            task (str): 'detect' or 'classify'. For multi-class detection, set 'detect'.
        """
        self.task = task
        self.nc = nc
        self.iou_thres = iou_thres

        # Instead of a single matrix, we store the "match" info for each image in a list
        # Each entry is a dict with:
        #   "gt_classes": shape (n_gt,)
        #   "det_classes": shape (n_det,)
        #   "det_confs": shape (n_det,)
        #   "matches": shape (m, 3) with columns = (gt_index, det_index, iou)
        # This is enough to rebuild confusion matrix for any conf threshold
        self.matches_per_image = []

    def accumulate_matches(self, detections, gt_bboxes, gt_cls):
        """
        Accumulate all detection-vs-gt matches for a single image, without confidence filtering.

        Args:
            detections (tensor): shape (n_det, 6 or 7). Format: (x1, y1, x2, y2, conf, class[, angle])
            gt_bboxes (tensor): shape (n_gt, 4 or 5). Ground-truth boxes
            gt_cls (tensor): shape (n_gt,). Ground-truth classes
        """
        # If no GT, store detections anyway (they become potential false positives for any conf>0)
        if gt_cls.shape[0] == 0:
            if detections is not None and len(detections) > 0:
                # We store them with no matches
                self.matches_per_image.append({
                    "gt_classes": torch.zeros((0,), dtype=torch.int),
                    "det_classes": detections[:, 5].int().cpu() if len(detections.shape) > 1 else [],
                    "det_confs": detections[:, 4].cpu() if len(detections.shape) > 1 else [],
                    "matches": np.zeros((0, 3))
                })
            return

        # If no detections
        if detections is None or len(detections) == 0:
            self.matches_per_image.append({
                "gt_classes": gt_cls.int().cpu(),
                "det_classes": torch.zeros((0,), dtype=torch.int),
                "det_confs": torch.zeros((0,)),
                "matches": np.zeros((0, 3))
            })
            return

        # Otherwise, we do IoU matching
        det_confs = detections[:, 4]
        det_classes = detections[:, 5].int()
        is_obb = (detections.shape[1] == 7 and gt_bboxes.shape[1] == 5)
        # Box iou
        if is_obb:
            from math import sqrt
            # Suppose you have a suitable OBB IoU function: batch_probiou
            iou = batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
        else:
            iou = box_iou(gt_bboxes, detections[:, :4])

        # Let's find all iou > self.iou_thres
        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            # Each matched pair is (gt_idx, det_idx, iou_val)
            temp_matches = torch.cat([torch.stack(x, 1), iou[x[0], x[1]][:, None]], dim=1)
            matches = temp_matches.cpu().numpy()

            # We apply the same matching logic from YOLO: highest iou gets the match
            matches = matches[matches[:, 2].argsort()[::-1]]  # sort by iou descending
            # Unique detection indices
            _, unique_det_idx = np.unique(matches[:, 1], return_index=True)
            matches = matches[unique_det_idx]
            # Unique GT
            matches = matches[matches[:, 2].argsort()[::-1]]
            _, unique_gt_idx = np.unique(matches[:, 0], return_index=True)
            matches = matches[unique_gt_idx]
        else:
            matches = np.zeros((0, 3))

        # Store results for later
        self.matches_per_image.append({
            "gt_classes": gt_cls.int().cpu(),
            "det_classes": det_classes.cpu(),
            "det_confs": det_confs.cpu(),
            "matches": matches
        })

    def compute_matrix(self, conf_thres=0.25):
        """
        Given a confidence threshold, compute and return the confusion matrix.

        The returned matrix is (nc+1, nc+1) in 'detect' mode:
            rows = predicted (including background as last row)
            cols = ground-truth (including background as last col)
        """
        # Initialize matrix
        if self.task == "detect":
            matrix = np.zeros((self.nc + 1, self.nc + 1), dtype=np.int32)
        else:
            matrix = np.zeros((self.nc, self.nc), dtype=np.int32)

        for item in self.matches_per_image:
            gt_classes = item["gt_classes"].numpy()
            det_classes = item["det_classes"].numpy()
            det_confs = item["det_confs"].numpy()
            matches = item["matches"]  # shape (m,3): [gt_idx, det_idx, iou]

            if len(gt_classes) == 0:
                # no ground-truth → all detections are false positives
                # but only for those with conf >= conf_thres
                if len(det_classes) > 0:
                    above = det_confs >= conf_thres
                    for dc in det_classes[above]:
                        if self.task == "detect":
                            matrix[dc, self.nc] += 1  # predicted class -> background col
                        else:
                            matrix[dc, 0] += 1
                continue

            if len(det_classes) == 0:
                # no detections → all gt are missed (FN)
                for gc in gt_classes:
                    if self.task == "detect":
                        matrix[self.nc, gc] += 1  # background row -> gt class
                    else:
                        matrix[gc, gc] += 1  # for classification tasks
                continue

            # We have both GT and detections
            # For confusion matrix, filter detections by conf_thres
            used_det = np.zeros_like(det_classes, dtype=bool)
            # Convert matches to int
            matches = matches.astype(int, copy=False)

            for gt_idx, det_idx, _ in matches:
                # If that detection has conf >= threshold
                if det_confs[det_idx] >= conf_thres and not used_det[det_idx]:
                    pred_c = det_classes[det_idx]
                    true_c = gt_classes[gt_idx]
                    if self.task == "detect":
                        matrix[pred_c, true_c] += 1
                    else:
                        matrix[pred_c, true_c] += 1
                    used_det[det_idx] = True
                # If conf < threshold or already used, it won't count as TP

            # All ground-truth that did not get matched above → FN
            # The matched GT indices are
            matched_gt = matches[:, 0]
            matched_gt_unique = np.unique(matched_gt)
            # We want the GT indices that did not match at all or matched with a detection < conf_thres
            unmatched_gt = list(set(range(len(gt_classes))) - set(matched_gt_unique))
            # For matched GT, check if the detection used had conf < threshold
            for g in matched_gt_unique:
                # which detection was used
                sub = matches[matches[:, 0] == g]
                # g can match multiple detections, but only 1 used if any
                used = any(
                    (det_confs[d_idx] >= conf_thres and not used_det[d_idx] == False)
                    for _, d_idx, __ in sub
                )
                if not used:
                    unmatched_gt.append(g)

            for g in unmatched_gt:
                gc = gt_classes[g]
                if self.task == "detect":
                    matrix[self.nc, gc] += 1
                else:
                    # for classification tasks, typically no background row
                    # but you can adapt logic
                    pass

            # Detections that are above conf but not matched → FP
            for d_idx, dc in enumerate(det_classes):
                if det_confs[d_idx] >= conf_thres and not used_det[d_idx]:
                    if self.task == "detect":
                        matrix[dc, self.nc] += 1
                    else:
                        # classification tasks, increment matrix[dc, ???]
                        pass

        return matrix

# ---------------- Utility IoU for boxes ---------------

def box_iou(box1, box2, eps=1e-7):
    """
    Standard box IoU.
    box1 shape: (N,4), box2 shape: (M,4) in xyxy
    """
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def batch_probiou(obb1, obb2, eps=1e-7):
    """
    Example oriented bounding box IoU.
    Replace with your actual OBB IoU implementation if desired.
    """
    # For demonstration, fallback to naive IoU on xyxy
    # Typically you'd implement the actual formula from the YOLO source.
    # Just returning standard box_iou to keep code consistent
    return box_iou(obb1[:, :4], obb2[:, :4], eps=eps)
