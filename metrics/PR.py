
import numpy as np
from shapely.geometry import box
import cv2
import matplotlib.pyplot as plt


# Option 1: Precision-Recall Curve
# Option 2: Precision vs IoU Threshold
# Option 3: Precision, Recall, and F1-score vs IoU Threshold

option = 1

# Rutas a los archivos de ground truth y predicciones
GT_PATH = "/mnt/work/users/marc.corretge/GT/ant_subset_1-024_GT.txt"
PRED_PATH = r"E:\TFG\manualTrackAnnotation\noSAHI\ant_subset_1-024_Pol_MOT.txt"
dire_of_frames = "/mnt/work/users/marc.corretge/GT/frames/"

iou_thresholds = np.arange(0.1, 1.1, 0.2)


#  Final declaracio de variables
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#  Inici del codi




if option == 1:

    # Cargar los datos
    GT = np.genfromtxt(GT_PATH, delimiter=',')
    PRED = np.genfromtxt(PRED_PATH, delimiter=',')

    # Obtener los frames únicos
    frames = np.unique(GT[:, 0])
    min_frame = int(min(frames))
    max_frame = int(max(frames))

    # Listas para almacenar los resultados
    precision_list = []
    recall_list = []

    # Función para calcular los valores de TP, FP, FN
    def calculate_metrics(iou_threshold):
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for frame in range(min_frame, max_frame + 1):
            img = cv2.imread(dire_of_frames + "\\frame" + str(frame).zfill(4) + ".png")
            list_done = []
            tp = 0
            fp = 0
            fn = 0

            listOfTP = []
            listOfFP = []
            listOfFN = []

            # Filtrar bboxes para el frame actual
            mask_gt = GT[GT[:, 0] == frame]
            mask_pred = PRED[PRED[:, 0] == frame]

            # Comparar todos los bboxes del GT con todos los bboxes de las predicciones
            for mot_gt in mask_gt:
                gt_bbox = box(mot_gt[2], mot_gt[3], mot_gt[2] + mot_gt[4], mot_gt[3] + mot_gt[5])
                max_IoU = 0
                best_match_pred = None
                matched = False

                for mot_pred in mask_pred:
                    pred_bbox = box(mot_pred[2], mot_pred[3], mot_pred[2] + mot_pred[4], mot_pred[3] + mot_pred[5])

                    # Calcular IoU
                    if gt_bbox.intersects(pred_bbox):
                        IoU = gt_bbox.intersection(pred_bbox).area / gt_bbox.union(pred_bbox).area

                        if IoU > max_IoU:
                            max_IoU = IoU
                            best_match_pred = mot_pred

                        if IoU >= iou_threshold:
                            tp += 1
                            list_done.append(mot_pred[1])
                            matched = True
                            listOfTP.append((mot_gt, mot_pred, IoU))
                            break

                if not matched:
                    fn += 1
                    listOfFN.append((mot_gt, best_match_pred, max_IoU))

            # Determinar FP
            for mot_pred in mask_pred:
                if mot_pred[1] not in list_done:
                    fp += 1
                    listOfFP.append((None, mot_pred, 0))

            # Acumular contadores
            total_tp += tp
            total_fp += fp
            total_fn += fn

        return total_tp, total_fp, total_fn

    # Calcular métricas para cada umbral de IoU
    for iou_threshold in iou_thresholds:
        tp, fp, fn = calculate_metrics(iou_threshold)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
        print(f'IoU Threshold={iou_threshold:.2f}: TP={tp}, FP={fp}, FN={fn}, Precision={precision:.2f}, Recall={recall:.2f}')

    # Generar la curva de precisión-recall
    plt.figure(figsize=(10, 5))
    plt.plot(recall_list, precision_list, marker='o')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.show()





    plt.subplot(1, 2, 1)
    plt.plot(iou_thresholds, precision_list, marker='o')
    plt.title('Precision vs IoU Threshold')
    plt.xlabel('IoU Threshold')
    plt.ylabel('Precision')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(iou_thresholds, recall_list, marker='o')
    plt.title('Recall vs IoU Threshold')
    plt.xlabel('IoU Threshold')
    plt.ylabel('Recall')
    plt.grid(True)

    plt.tight_layout()
    plt.show()










if option == 2:

    # Cargar los datos
    GT = np.genfromtxt(GT_PATH, delimiter=',')
    PRED = np.genfromtxt(PRED_PATH, delimiter=',')

    # Obtener los frames únicos
    frames = np.unique(GT[:, 0])
    min_frame = int(min(frames))
    max_frame = int(max(frames))

    # Listas para almacenar precisión y recall
    precision_list = []
    recall_list = []

    # Iterar sobre diferentes valores de umbral de IoU
    for iou_threshold in iou_thresholds:
        # Inicializar contadores globales
        total_tp = 0
        total_fp = 0
        total_fn = 0

        # Iterar sobre cada frame
        for frame in range(min_frame, max_frame + 1):
            list_done = []
            tp = 0
            fp = 0
            fn = 0

            # Filtrar bboxes para el frame actual
            mask_gt = GT[GT[:, 0] == frame]
            mask_pred = PRED[PRED[:, 0] == frame]

            # Comparar todos los bboxes del GT con todos los bboxes de las predicciones
            for mot_gt in mask_gt:
                gt_bbox = box(mot_gt[2], mot_gt[3], mot_gt[2] + mot_gt[4], mot_gt[3] + mot_gt[5])
                max_IoU = 0
                best_match_pred = None
                matched = False

                for mot_pred in mask_pred:
                    pred_bbox = box(mot_pred[2], mot_pred[3], mot_pred[2] + mot_pred[4], mot_pred[3] + mot_pred[5])

                    # Calcular IoU
                    if gt_bbox.intersects(pred_bbox):
                        IoU = gt_bbox.intersection(pred_bbox).area / gt_bbox.union(pred_bbox).area

                        if IoU > max_IoU:
                            max_IoU = IoU
                            best_match_pred = mot_pred

                        if IoU >= iou_threshold:
                            tp += 1
                            list_done.append(mot_pred[1])
                            matched = True
                            break

                if not matched:
                    fn += 1

            # Determinar FP
            for mot_pred in mask_pred:
                if mot_pred[1] not in list_done:
                    fp += 1

            # Acumular contadores
            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Calcular precisión y recall para el umbral actual
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

        # Agregar precisión y recall a las listas
        precision_list.append(precision)
        recall_list.append(recall)

    # Graficar la curva Precision-Recall
    plt.plot(recall_list, precision_list, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()












if option == 3:

    
    # Cargar los datos
    GT = np.genfromtxt(GT_PATH, delimiter=',')
    PRED = np.genfromtxt(PRED_PATH, delimiter=',')
    
    # Obtener los frames únicos
    frames = np.unique(GT[:, 0])
    min_frame = int(min(frames))
    max_frame = int(max(frames))
    
    # Listas para almacenar precisión, recall y F1-score
    precision_list = []
    recall_list = []
    f1_list = []
    
    # Iterar sobre diferentes valores de umbral de IoU    
    for iou_threshold in iou_thresholds:
        # Inicializar contadores globales
        total_tp = 0
        total_fp = 0
        total_fn = 0
    
        # Iterar sobre cada frame
        for frame in range(min_frame, max_frame + 1):
            list_done = []
            tp = 0
            fp = 0
            fn = 0
    
            # Filtrar bboxes para el frame actual
            mask_gt = GT[GT[:, 0] == frame]
            mask_pred = PRED[PRED[:, 0] == frame]
    
            # Comparar todos los bboxes del GT con todos los bboxes de las predicciones
            for mot_gt in mask_gt:
                gt_bbox = box(mot_gt[2], mot_gt[3], mot_gt[2] + mot_gt[4], mot_gt[3] + mot_gt[5])
                max_IoU = 0
                best_match_pred = None
                matched = False
    
                for mot_pred in mask_pred:
                    pred_bbox = box(mot_pred[2], mot_pred[3], mot_pred[2] + mot_pred[4], mot_pred[3] + mot_pred[5])
    
                    # Calcular IoU
                    if gt_bbox.intersects(pred_bbox):
                        IoU = gt_bbox.intersection(pred_bbox).area / gt_bbox.union(pred_bbox).area
    
                        if IoU > max_IoU:
                            max_IoU = IoU
                            best_match_pred = mot_pred
    
                        if IoU >= iou_threshold:
                            tp += 1
                            list_done.append(mot_pred[1])
                            matched = True
                            break
                        
                if not matched:
                    fn += 1
    
            # Determinar FP
            for mot_pred in mask_pred:
                if mot_pred[1] not in list_done:
                    fp += 1
    
            # Acumular contadores
            total_tp += tp
            total_fp += fp
            total_fn += fn
    
        # Calcular precisión, recall y F1-score para el umbral actual
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
        # Agregar precisión, recall y F1-score a las listas
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1_score)
    
    # Graficar precisión, recall y F1-score en función del umbral de IoU
    plt.plot(iou_thresholds, precision_list, label='Precision')
    plt.plot(iou_thresholds, recall_list, label='Recall')
    plt.plot(iou_thresholds, f1_list, label='F1-score')
    plt.xlabel('IoU Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1-score vs IoU Threshold')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()
