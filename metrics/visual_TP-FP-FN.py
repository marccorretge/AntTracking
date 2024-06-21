import numpy as np
from shapely.geometry import box
import cv2

# Rutas a los archivos de ground truth y predicciones
GT_PATH = "/mnt/work/users/marc.corretge/GT/ant_subset_1-024_GT.txt"
PRED_PATH = r"E:\TFG\manualTrackAnnotation\noSAHI\ant_subset_1-024_Pol_MOT.txt"
dire_of_frames = "/mnt/work/users/marc.corretge/GT/ant_subset_1-024_GT.txt"

# Umbral de IoU
iou_threshold = 0.5


#  Final declaracio de variables
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#  Inici del codi




# Cargar los datos
GT = np.genfromtxt(GT_PATH, delimiter=',')
PRED = np.genfromtxt(PRED_PATH, delimiter=',')



# Obtener los frames únicos
frames = np.unique(GT[:, 0])
min_frame = int(min(frames))
max_frame = int(max(frames))

# Inicializar contadores globales
total_tp = 0
total_fp = 0
total_fn = 0

# Iterar sobre cada frame
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

    # Dibujar los bboxes en la imagen
    for mot_gt, mot_pred, IoU in listOfTP:
        img = cv2.rectangle(img, (int(mot_gt[2]), int(mot_gt[3])), (int(mot_gt[2] + mot_gt[4]), int(mot_gt[3] + mot_gt[5])), (0, 255, 0), 4)
        img = cv2.rectangle(img, (int(mot_pred[2]), int(mot_pred[3])), (int(mot_pred[2] + mot_pred[4]), int(mot_pred[3] + mot_pred[5])), (0, 255, 0), 2)
        img = cv2.putText(img, f'IoU: {IoU:.2f}', (int(mot_pred[2]), int(mot_pred[3] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    for _, mot_pred, IoU in listOfFP:
        img = cv2.rectangle(img, (int(mot_pred[2]), int(mot_pred[3])), (int(mot_pred[2] + mot_pred[4]), int(mot_pred[3] + mot_pred[5])), (0, 0, 255), 4)
        img = cv2.putText(img, f'IoU: {IoU:.2f}', (int(mot_pred[2]), int(mot_pred[3] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    for mot_gt, mot_pred, IoU in listOfFN:
        img = cv2.rectangle(img, (int(mot_gt[2]), int(mot_gt[3])), (int(mot_gt[2] + mot_gt[4]), int(mot_gt[3] + mot_gt[5])), (255, 0, 0), 4)
        img = cv2.putText(img, f'IoU: {IoU:.2f}', (int(mot_pred[2]), int(mot_pred[3] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        if mot_pred is not None:
            img = cv2.rectangle(img, (int(mot_pred[2]), int(mot_pred[3])), (int(mot_pred[2] + mot_pred[4]), int(mot_pred[3] + mot_pred[5])), (255, 0, 0), 2)

    print(f'Frame {frame}: TP={tp}, FP={fp}, FN={fn}')
    # Redimensionar y mostrar la imagen
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    cv2.imshow("Frame", img)
    cv2.waitKey(15)
    #cv2.destroyAllWindows()

    

# Mostrar resultados totales
print(f'Total TP={total_tp}, Total FP={total_fp}, Total FN={total_fn}')
