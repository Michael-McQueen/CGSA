import numpy as np

def print_per_class_ap50(coco_eval, coco_api):

    precisions = coco_eval.eval['precision']  # [T, R, K, A, M]
    cat_ids = coco_eval.params.catIds
    names = [coco_api.cats[cid]["name"] for cid in cat_ids]

    area_idx = 0       # all area
    max_det_idx = 2    # 100 dets

    # AP50
    iou_thr_idx_50 = 0
    per_class_ap50 = precisions[iou_thr_idx_50, :, :, area_idx, max_det_idx]  # [R, K]
    ap50 = np.mean(per_class_ap50, axis=0)  # [K]

    # AP@[.50:.95]
    per_class_ap_all = precisions[:, :, :, area_idx, max_det_idx]  # [T, R, K]
    ap_all = np.mean(per_class_ap_all, axis=(0, 1))  # [K]

    print("\nPer-class AP50:")
    for n, v in zip(names, ap50):
        if v < 0:
            print(f"{n:>20}:   n/a")
        else:
            print(f"{n:>20}: {v * 100:6.2f}")
    
    print("\nPer-class AP@[.50:.95]:")
    for n, v in zip(names, ap_all):
        if v < 0:
            print(f"{n:>20}:   n/a")
        else:
            print(f"{n:>20}: {v * 100:6.2f}")
    
    print("-" * 40)

    return {
        n: {
            "ap50": None if ap50[i] < 0 else float(ap50[i]),
            "ap":   None if ap_all[i] < 0 else float(ap_all[i])
        }
        for i, n in enumerate(names)
    }
