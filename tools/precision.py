import json
import os
import numpy as np
import cv2

def load_predictions(predictions_file):
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    return predictions

def load_ground_truths(gt_dir, image_ids):
    ground_truths = []
    for image_id in image_ids:
        gt_file = os.path.join(gt_dir, f"{image_id}.txt")
        if os.path.exists(gt_file):
            img = cv2.imread(os.path.join('dataset/images/test', f"{image_id}.jpg"))

            img_width, img_height = img.shape[1], img.shape[0]
            with open(gt_file, 'r') as f:
                gt_boxes = []
                for line in f:
                    parts = line.strip().split()
                    class_id, x_center, y_center, width, height = map(float, parts)

                    # Because yolo didnt normalize the predictions :/
                    x_center *= img_width
                    y_center *= img_height
                    width *= img_width
                    height *= img_height

                    gt_boxes.append([x_center - width / 2, y_center - height / 2, width, height])
                ground_truths.append({"image_id": image_id, "bbox": gt_boxes})
    return ground_truths

def iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Boxes are in the format [x, y, width, height].
    """

    if(len(box1) == 0 or len(box2) == 0):
        return 0
    
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    area_box1 = w1 * h1
    area_box2 = w2 * h2
    
    union_area = area_box1 + area_box2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0

def calculate_tp_fn(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate the number of true positives (TP) and false negatives (FN) 
    based on the model predictions and ground truth boxes.
    """
    true_positives = 0
    false_negatives = 0
    matched_gt = set() 
    
    for pred in predictions:
        pred_bbox = pred['bbox']
        pred_score = pred['score']
        pred_image_id = pred['image_id']
        
        image_gt_boxes = [gt['bbox'] for gt in ground_truths if gt['image_id'] == pred_image_id]
        
        best_iou = 0
        for gt_arr in image_gt_boxes:
            for gt_bbox in gt_arr:
                iou_score = iou(pred_bbox, gt_bbox)
                best_iou = max(best_iou, iou_score)

        
                # If IoU is above threshold, it's a true positive
                if best_iou >= iou_threshold and pred_score > 0.01:  #  confidence threshold for TP
                    true_positives += 1
                    matched_gt.add(pred_image_id)
    
    # False negatives: ground truth that wasn't matched
    for gt in ground_truths:
        if gt['image_id'] not in matched_gt:
            false_negatives += 1
    
    return true_positives, false_negatives


predictions_file = 'runs/detect/test/predictions.json'
gt_dir = 'dataset/labels/test'  # Ground truths

predictions = load_predictions(predictions_file)

image_ids = set(pred['image_id'] for pred in predictions)

ground_truths = load_ground_truths(gt_dir, image_ids)

tp, fn = calculate_tp_fn(predictions, ground_truths, iou_threshold=0.3)
print(f"True Positives: {tp}, False Negatives: {fn}")