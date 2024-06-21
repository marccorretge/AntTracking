"""
author: Timothy C. Arlen
date: 28 Feb 2018

Calculate Mean Average Precision (mAP) for a set of bounding boxes corresponding to specific
image Ids. Usage:

> python calculate_mean_ap.py

Will display a plot of precision vs recall curves at 10 distinct IoU thresholds as well as output
summary information regarding the average precision and mAP scores.

NOTE: Requires the files `ground_truth_boxes.json` and `predicted_boxes.json` which can be
downloaded fromt this gist.
"""

# https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734

from __future__ import absolute_import, division, print_function

from copy import deepcopy
import json
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import cv2

sns.set_style('white')
sns.set_context('poster')

COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']


def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box

    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

    Returns:
        float: value of the IoU for the two boxes.

    Raises:
        AssertionError: if the box is obviously malformed
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou


def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.

    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.

    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """

    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou_individual(pred_box, gt_box)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)

    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}


def calc_precision_recall(img_results):
    """Calculates precision and recall from the set of images

    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }

    Returns:
        tuple: of floats of (precision, recall)
    """
    true_pos = 0; false_pos = 0; false_neg = 0
    for _, res in img_results.items():
        true_pos += res['true_pos']
        false_pos += res['false_pos']
        false_neg += res['false_neg']

    try:
        precision = true_pos/(true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_pos/(true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)

def get_model_scores_map(pred_boxes):
    """Creates a dictionary of from model_scores to image ids.

    Args:
        pred_boxes (dict): dict of dicts of 'boxes' and 'scores'

    Returns:
        dict: keys are model_scores and values are image ids (usually filenames)

    """
    model_scores_map = {}
    for img_id, val in pred_boxes.items():
        for score in val['scores']:
            if score not in model_scores_map.keys():
                model_scores_map[score] = [img_id]
            else:
                model_scores_map[score].append(img_id)
    return model_scores_map

def get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=0.5):
    """Calculates average precision at given IoU threshold.

    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (list of list of floats): list of locations of predicted
            objects as [xmin, ymin, xmax, ymax]
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.

    Returns:
        dict: avg precision as well as summary info about the PR curve

        Keys:
            'avg_prec' (float): average precision for this IoU threshold
            'precisions' (list of floats): precision value for the given
                model_threshold
            'recall' (list of floats): recall value for given
                model_threshold
            'models_thrs' (list of floats): model threshold value that
                precision and recall were computed for.
    """
    model_scores_map = get_model_scores_map(pred_boxes)
    sorted_model_scores = sorted(model_scores_map.keys())

    # Sort the predicted boxes in descending order (lowest scoring boxes first):
    for img_id in pred_boxes.keys():
        arg_sort = np.argsort(pred_boxes[img_id]['scores'])
        pred_boxes[img_id]['scores'] = np.array(pred_boxes[img_id]['scores'])[arg_sort].tolist()
        pred_boxes[img_id]['boxes'] = np.array(pred_boxes[img_id]['boxes'])[arg_sort].tolist()

    pred_boxes_pruned = deepcopy(pred_boxes)

    precisions = []
    recalls = []
    model_thrs = []
    img_results = {}
    # Loop over model score thresholds and calculate precision, recall
    for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
        # On first iteration, define img_results for the first time:
        img_ids = gt_boxes.keys() if ithr == 0 else model_scores_map[model_score_thr]
        for img_id in img_ids:
            gt_boxes_img = gt_boxes[img_id]
            box_scores = pred_boxes_pruned[img_id]['scores']
            start_idx = 0
            for score in box_scores:
                if score <= model_score_thr:
                    pred_boxes_pruned[img_id]
                    start_idx += 1
                else:
                    break

            # Remove boxes, scores of lower than threshold scores:
            pred_boxes_pruned[img_id]['scores'] = pred_boxes_pruned[img_id]['scores'][start_idx:]
            pred_boxes_pruned[img_id]['boxes'] = pred_boxes_pruned[img_id]['boxes'][start_idx:]

            # Recalculate image results for this image
            img_results[img_id] = get_single_image_results(
                gt_boxes_img, pred_boxes_pruned[img_id]['boxes'], iou_thr)

        prec, rec = calc_precision_recall(img_results)
        precisions.append(prec)
        recalls.append(rec)
        model_thrs.append(model_score_thr)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(recalls >= recall_level).flatten()
            prec = max(precisions[args])
        except ValueError:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)

    return {
        'avg_prec': avg_prec,
        'precisions': precisions,
        'recalls': recalls,
        'model_thrs': model_thrs}


def plot_pr_curve(
    precisions, recalls, category='Person', label=None, color=None, ax=None):
    """Simple plotting helper function"""

    if ax is None:
        plt.figure(figsize=(10,8))
        ax = plt.gca()

    if color is None:
        color = COLORS[0]
    ax.scatter(recalls, precisions, label=label, s=20, color=color)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    #ax.set_title('Precision-Recall curve for {}'.format(category))
    ax.set_title('Precision-Recall curve')
    ax.set_xlim([0.0,1.3])
    ax.set_ylim([0.0,1.2])
    return ax




def prepareData(GT, PRED, ground_truth, images_optional, detection_results):
    os.makedirs(ground_truth, exist_ok=True)
    os.makedirs(images_optional, exist_ok=True)
    os.makedirs(detection_results, exist_ok=True)

    # PROCESS GROUND TRUTH
    # delete the content of the folder
    for file in os.listdir(ground_truth):
        os.remove(os.path.join(ground_truth, file))

    # PROCESS PREDICTIONS
    # delete the content of the folder
    for file in os.listdir(detection_results):
        os.remove(os.path.join(detection_results, file))



    gt = np.genfromtxt(GT, delimiter=',', dtype=float, encoding=None)
    pred = np.genfromtxt(PRED, delimiter=',', dtype=float, encoding=None)

    if gt[0,0] == 'frame':
            gt = gt[1:]

    if pred[0,0] == 'frame':
        pred = pred[1:]


    frames = gt[:,0].astype(int)
    frames = np.unique(frames)
    minFrame = frames[0]
    maxFrame = frames[-1]

    frames = pred[:,0].astype(int)
    frames = np.unique(frames)

    for frame in range(minFrame, maxFrame+1):

        # read image
        #imgPath = "E:/TFG/code/EVALs/mAP/input/images-optional_/frame" + str(frame).zfill(4) + ".png"
        #img = cv2.imread(imgPath)

        with open(os.path.join(ground_truth, f"frame{frame:04d}.txt"), "w") as file:
        #with open(os.path.join("E:\TFG\code\EVALs\mAP", f"frame{frame:04d}.txt"), "w") as file:
            mask = gt[gt[:,0].astype(int) == frame]
            # order by id
            mask = mask[np.argsort(mask[:,1])]

            for line in mask:
                #if frame == 112:
                #    print(line[1])
                class_id = str(int(line[1]))
                class_id = "Ant"
                x0 = float(line[2])
                y0 = float(line[3])
                w = float(line[4])
                h = float(line[5])
                x1 = x0 + w
                y1 = y0 + h
                file.write(f"{class_id} {x0} {y0} {x1} {y1}\n")
                #cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), 2)
                #cv2.putText(img, class_id, (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)



        with open(os.path.join(detection_results, f"frame{frame:04d}.txt"), "w") as file:
            mask = pred[pred[:,0].astype(int) == frame]
            # order by id
            mask = mask[np.argsort(mask[:,1])]

            for line in mask:
                class_id = str(int(line[1]))
                class_id = "Ant"
                x0 = float(line[2])
                y0 = float(line[3])
                w = float(line[4])
                h = float(line[5])
                x1 = x0 + w
                y1 = y0 + h
                score = line[6]
                #if class_id == '47':
                #    class_id = '32'
                #elif class_id == '32':
                #    class_id = '47'
                file.write(f"{class_id} {score} {x0} {y0} {x1} {y1}\n")
                #cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
                #cv2.putText(img, class_id, (60+int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)


        #img_resized = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
        #cv2.imshow("img", img_resized)
        #cv2.waitKey(15)

        # call main.py with the arguments
    print(PRED)




if __name__ == "__main__":

            
    GT = r"E:\TFG\manualTrackAnnotation\GT\ant_subset_1-024_GT_noID.txt" # Path to the ground truth file
    PRED = r"E:\TFG\manualTrackAnnotation\SAHI\ant_subset_1-024_Marc_maxAge2_MOT.csv" # Path to the predictions file (detections)
    ground_truth = "E:\TFG\code\EVALs\mAP\input\ground-truth" # Path to the ground truth folder
    images_optional = "E:\TFG\code\EVALs\mAP\input\images-optional" # Path to the folder with the images (not used)
    detection_results = "E:\TFG\code\EVALs\mAP\input\detection-results" # Path to the folder with the detection results
    prepareData(GT, PRED, ground_truth, images_optional, detection_results) # Prepare the data for the mAP calculation

    iou_thr = 0.5

    #  Final declaracio de variables
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #  Inici del codi



    ######################################################
    ################        GT          ##################
    ######################################################
    # Definir el directorio que contiene los archivos
    input_directory = ground_truth  # Reemplaza esto con la ruta real a tu directorio
    output_file = 'ground_truth_boxes.json'

    # Inicializar el diccionario para almacenar los datos
    ground_truth_boxes = {}
    
    # Recorrer cada archivo en el directorio
    for filename in os.listdir(input_directory):
        if filename.endswith('.txt'):  # Asegúrate de que solo procesas archivos .txt
            frame_number = filename.split('.')[0]  # Obtener el número de frame desde el nombre del archivo
            frame_key = f"{frame_number}.jpg"  # Crear la clave del diccionario
            #if int(frame_number[5:]) > 2 and int(frame_number[5:]) < 10:
            #    pass
            #else:
            #    continue
            # Abrir y leer el archivo
            with open(os.path.join(input_directory, filename), 'r') as file:
                boxes = []
                cont = 0
                for line in file:
                    #cont += 1   
                    #if cont == 3:
                    #    #break
                    #    pass
                    parts = line.strip().split(' ')
                    #if len(parts) >= 5:
                    x0 = int(float(parts[1]))
                    y0 = int(float(parts[2]))
                    x1 = int(float(parts[3]))
                    y1 = int(float(parts[4]))
                    box = [x0, y0, x1, y1]
                    boxes.append(box)

                ground_truth_boxes[frame_key] = boxes

    # Guardar los datos en un archivo JSON
    with open(output_file, 'w') as json_file:
        json.dump(ground_truth_boxes, json_file)

    print(f"Datos guardados en {output_file}")


    ######################################################
    ############        DETECTIONS          ##############
    ######################################################
    # Definir el directorio que contiene los archivos
    input_directory = detection_results  # Reemplaza esto con la ruta real a tu directorio
    output_file = 'predicted_boxes.json'
    # Inicializar el diccionario para almacenar los datos
    # Diccionario para almacenar los datos en el formato requerido
    predicted_boxes = {}
    
    # Recorrer todos los archivos en el directorio
    for nombre_archivo in os.listdir(input_directory):
        if nombre_archivo.endswith('.txt'):
            # Extraer el número de frame del nombre del archivo
            numero_frame = int(nombre_archivo[5:9])
            # Generar el nombre de la imagen correspondiente
            nombre_imagen = f"frame{numero_frame:04d}.jpg"                
            #if int(numero_frame) > 2 and int(numero_frame) < 10:
            #    pass
            #else:
            #    continue
            # Inicializar listas para las cajas y las puntuaciones
            boxes = []
            scores = []
            
            # Leer el archivo de texto
            ruta_archivo = os.path.join(input_directory, nombre_archivo)
            with open(ruta_archivo, 'r') as archivo:
                cont = 0
                for linea in archivo:
                    #cont += 1   
                    #if cont == 3:
                    #    #break
                    #    pass
                    # Separar los valores por coma
                    valores = linea.strip().split(' ')
                    # Extraer las coordenadas y la puntuación
                    left = int(float(valores[2]))
                    top = int(float(valores[3]))
                    width = int(float(valores[4]))
                    height = int(float(valores[5]))
                    try:
                        score = float(valores[6])
                    except:
                        score = np.random.rand()
                        #score = 1.0
                    # Calcular las coordenadas de la caja
                    x1 = left
                    y1 = top
                    x2 = width
                    y2 = height
                    # Agregar la caja y la puntuación a las listas
                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
            
            # Agregar los datos al diccionario
            predicted_boxes[nombre_imagen] = {"boxes": boxes, "scores": scores}


        
    
    # Guardar el diccionario en un archivo JSON
    with open('predicted_boxes.json', 'w') as json_file:
        json.dump(predicted_boxes, json_file)



    with open('ground_truth_boxes.json') as infile:
        gt_boxes = json.load(infile)

    with open('predicted_boxes.json') as infile:
        pred_boxes = json.load(infile)

    # Runs it for one IoU threshold
    start_time = time.time()
    data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
    end_time = time.time()
    print('Single IoU calculation took {:.4f} secs'.format(end_time - start_time))
    print('avg precision: {:.4f}'.format(data['avg_prec']))

    start_time = time.time()
    ax = None
    avg_precs = []
    iou_thrs = []
    for idx, iou_thr in enumerate(np.linspace(0.5, 0.95, 10)):
        data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
        avg_precs.append(data['avg_prec'])
        iou_thrs.append(iou_thr)

        precisions = data['precisions']
        recalls = data['recalls']
        ax = plot_pr_curve(
            precisions, recalls, label='{:.2f}'.format(iou_thr), color=COLORS[idx*2], ax=ax)

    # prettify for printing:
    avg_precs = [float('{:.4f}'.format(ap)) for ap in avg_precs]
    iou_thrs = [float('{:.4f}'.format(thr)) for thr in iou_thrs]
    print('map: {:.2f}'.format(100*np.mean(avg_precs)))
    print('avg precs: ', avg_precs)
    print('iou_thrs:  ', iou_thrs)
    plt.legend(loc='upper right', title='IOU Thr', frameon=True)
    for xval in np.linspace(0.0, 1.0, 11):
        plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')
    end_time = time.time()
    print('\nPlotting and calculating mAP takes {:.4f} secs'.format(end_time - start_time))
    plt.show()