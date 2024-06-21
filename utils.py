import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import shapely.geometry as sg
from shapely.ops import unary_union

color = [
        (255, 0, 0),     # Rojo
        (0, 255, 0),     # Verde
        (0, 0, 255),     # Azul
        (255, 255, 0),   # Amarillo
        (0, 255, 255),   # Cian
        (255, 0, 255),   # Magenta
        (255, 255, 255), # Blanco
        (100, 100, 100), # Gris oscuro
        (200, 200, 200), # Gris claro
        (128, 0, 0),     # Rojo oscuro
        (0, 128, 0),     # Verde oscuro
        (0, 0, 128),     # Azul oscuro
        (128, 128, 0),   # Amarillo oscuro
        (128, 0, 128),   # Magenta oscuro
        (0, 128, 128),   # Cian oscuro
        (128, 128, 128), # Gris medio
        (192, 192, 192), # Gris claro 2
        (64, 0, 0),      # Rojo muy oscuro
        (0, 64, 0)       # Verde muy oscuro
]



bright_colors = [
    (255, 0, 255),     # Azul
    (255, 0, 128),     # Azul
    (255, 0, 0),     # Verde
    (150, 0, 0),     # Verde
    (0, 150, 0),     # Verde
    (0, 255, 0),     # Verde
    (0, 0, 255),     # Azul
    (0, 0, 150),     # Azul
    (0, 0, 0),     # Rojo
    (0, 255, 150),     # Azul
    (157, 75, 0),     # Azul
    (0, 128, 255),     # Azul
    (160, 160, 160),     # Azul
]

bright_colors = [(r, g, b) for (b, g, r) in bright_colors]

color_black = (0, 0, 0)

def video2Frames(video_path, output_path):
    # Verifica que el directorio de salida exista, si no, lo crea
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Captura el video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error al abrir el video {video_path}")
        return
    
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        print("Frame ", frame_count)
        # Genera el nombre del archivo
        frame_filename = os.path.join(output_path, f"frame{frame_count:04d}.png")
    
        # Guarda el frame
        cv2.imwrite(frame_filename, frame)
               
    
    cap.release()
    print(f"Se han extraído {frame_count} frames.")


#########################################
#   Funció que ens permet retallar      #
#   un video en un interval de temps    #
#########################################
def cut_Vid(vid_path, start_time, end_time, output_path, fps=15):
    
    init_frame = start_time*fps
    end_frame = end_time*fps

    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()

    cap_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps,
                              (frame.shape[1], frame.shape[0]))
    
    frame_cnt = 0    
    
    while ret:
        if frame_cnt >= init_frame:
            cap_out.write(frame)

        if frame_cnt >= end_frame:
            break

        ret, frame = cap.read()
        frame_cnt += 1
    
    cap.release()
    cap_out.release()
    cv2.destroyAllWindows()
    

#cut_Vid(r"persona.mp4", 2, 5, r"persona_cutted.mp4")
    

############################################
#     Donat frame + bboxes, generar la     #
#   imatge de sortida + llista de bboxes   #
############################################
def process_bboxes(frame, print_bboxes, YOLO=False, SAHI=False):

    detections_MOT = []

    if SAHI:
        for result in frame.object_prediction_list:
            class_id = result.category.id
            class_name = result.category.name
            score = result.score.value
            area_bbox = result.bbox.area
            x1, y1, x2, y2 = result.bbox.minx, result.bbox.miny, result.bbox.maxx, result.bbox.maxy

            draw = ImageDraw.Draw(frame.image)
            font = ImageFont.truetype("arial.ttf", 12)
            color_ok = (0, 0, 255)
            color_ko = (255, 0, 0)        
            color_black = (0, 0, 0)

            detections_MOT.append([class_id, x1, y1, x2, y2, score, class_name])
            if print_bboxes:
                draw.text((x1, y1-12), str(np.round(score,1)), fill=color_black, font=font)

                if score > 0.5:            
                    draw.rectangle([x1, y1, x2, y2], outline=color_ok, width=2)

                #else:
                #    draw.rectangle([x1, y1, x2, y2], outline=color_ko, width=2)
        return frame, detections_MOT

    if YOLO:
        for r in frame[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r

            draw = ImageDraw.Draw(Image.fromarray(frame[0].orig_img))
            font = ImageFont.truetype("arial.ttf", 12)
            color_ok = (0, 0, 255)
            color_ko = (255, 0, 0)        
            color_black = (0, 0, 0)

            detections_MOT.append([class_id, x1, y1, x2, y2, score, frame[0].names[class_id]])

            if print_bboxes:
                draw.text((x1, y1-12), str(np.round(score,1)), fill=color_black, font=font)

            if score > 0.5:            
                draw.rectangle([x1, y1, x2, y2], outline=color_ok, width=2)

            else:
                draw.rectangle([x1, y1, x2, y2], outline=color_ko, width=2)

        return frame[0].orig_img, detections_MOT
    

############################################
#          Afegir bbox al MOT list         #
############################################

def add_bbox_2_MOT_Matrix(frame_cnt, bbox):
    MOT_Matrix = np.zeros((len(bbox), 5))
    for index, i in enumerate(bbox):
        class_id, x1, y1, x2, y2, score, class_name = i
        #MOT.append([frame_cnt, -1, x1, y1, x2-x1, y2-y1, score, class_id, -1])
        
        MOT_Matrix[index] = [x1, y1, x2, y2, score]



    return MOT_Matrix
    

############################################
#     Donat frame + bboxes, generar la     #
#            imatge de sortida             #
############################################
def print_bboxes(frame, list_to_print, threshold=0.5):


    for to_print in list_to_print:
        area = to_print[0]
        maxx = to_print[1]
        maxy = to_print[2]
        minx = to_print[3]
        miny = to_print[4]
        shift_x = to_print[5]
        shift_y = to_print[6]
        score = np.round(to_print[7],1)
        class_id = to_print[8]
        class_name = to_print[9]
        

        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype("arial.ttf", 12)
        color_ok = (0, 255, 0)
        color_ko = (0, 0, 255)        
        color_black = (0, 0, 0)

        draw.text((minx, miny-12), str(np.round(score,1)), fill=color_black, font=font)

        if score > threshold:            
            draw.rectangle([minx, miny, maxx, maxy], outline=color[class_id%len(color)], width=3)
        #else:
        #    draw.rectangle([minx, miny, maxx, maxy], outline=color_black, width=2)

    return frame



############################################
#              compare 2 videos            #
############################################
def compareVid(video1_path, video2_path, directory_out):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    #get cap1 width and height
    width1 = int(cap1.get(3))
    height1 = int(cap1.get(4))

    #get cap2 width and height
    width2 = int(cap2.get(3))
    height2 = int(cap2.get(4))

                  
    capout = cv2.VideoWriter(directory_out, cv2.VideoWriter_fourcc(*'MP4V'), cap1.get(cv2.CAP_PROP_FPS),
                            (width1+width2, int(np.max([height1, height2]))))
    
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    frame_cnt = 0
    while ret1 and ret2:
        if frame1.shape != frame2.shape:
            print(f"Frame {frame_cnt} is different")
            break
        else:
            frame = np.concatenate((frame1, frame2), axis=1)
            capout.write(frame)

        cv2.imshow("frame", frame)
        cv2.waitKey(25)
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        frame_cnt += 1
    
    
    cap1.release()
    cap2.release()
    capout.release()

    cv2.destroyAllWindows()

#compareVid(r"persona.mp4", r"persona_out.mp4", r"persona_out_compare.mp4")
    

############################################
#              plot paths lines            #
############################################
def plot_path(path):
    
    maxx = 0
    maxy = 0
    minx = 0
    miny = 0

    for i in path:
        for j in i:
            if j[2] > maxx:
                maxx = j[2]
            if j[3] > maxy:
                maxy = j[3]
            if j[0] < minx:
                minx = j[0]
            if j[1] < miny:
                miny = j[1]

    frame = 255 * np.ones((maxx-minx, maxy-miny, 3), dtype=np.uint8)

   

    for i in path:
        if len(i) > 1:
            for j in range(len(i)-1):
                cv2.line(frame, (i[j][0]+int((i[j][2]-i[j][0])/2), i[j][1]+int((i[j][3]-i[j][1])/2)), (i[j+1][0]+int((i[j+1][2]-i[j+1][0])/2), i[j+1][1]+int((i[j+1][3]-i[j+1][1])/2)), color=(0,0,0), thickness=2)
                    
    cv2.imshow("frame", frame)
    


############################################
#      convert lists of lists in matrix    #
############################################
def list2matrix(lists):
    matrix = np.zeros((len(lists), len(lists[0]), 3))
    for x,i in enumerate(lists):
        for y,j in enumerate(i):
            if j[0:8] != (-1,-1,-1,-1,-1,-1,-1,-1):
                matrix[x][y] = bright_colors[j[4]%len(bright_colors)]
            else:
                matrix[x][y] = color_black
    return np.array(matrix)


############################################
#   donat MOT + Video = show BB in Video   #
############################################
def displayVideoWithBBoxes(MOT_path, video_path, frameAndWait=True):
    cap = cv2.VideoCapture(video_path)
    
    # delete 1st line of MOT
    with open(MOT_path, 'r') as fin:
        data = fin.read().splitlines(True)
    
    if data[0].split(',')[0] == 'frame':
        MOT = data[1:]
    else:
        MOT = data       
    MOT = np.array([i.split(',') for i in MOT]).astype(float)
    

    # read Mot file from csv
    # S'HA DELIMINAR LA 1A LINEA DEL MOT
    #MOT = np.loadtxt(MOT_path, delimiter=',')
    
    frame_indices = MOT[:,0].astype(int)

    ret, frame = cap.read()
    frame_cnt = 0
    while ret:
        frame_cnt += 1
        mask = frame_indices == frame_cnt
        rows = MOT[mask]

        for row in rows:
            try:
                numFrame, id, x1, y1, w, h, score, x3d, y3d, z3d = row
            except:
                numFrame, id, x1, y1, w, h, score = row
            numFrame = int(numFrame)
            id = int(id)

            if x1 != -1 and y1 != -1 and w != -1 and h != -1:  
                if score < 0.75:
                    continue              
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x1+w), int(y1+h)), bright_colors[id%len(bright_colors)], 2)
                cv2.putText(frame, str(int(id)), (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color_black, 2)
                cv2.putText(frame, str(int(score*100)), (int(x1), int(y1+h)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color_black, 2)
                pass

        frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        cv2.imshow(str(video_path.split("\\")[-1].split(".")[0]), frame)
        if frameAndWait:
            # wait for key press
            # wait space key to continue
            k = cv2.waitKey(0)
            if k == 32:
                cv2.destroyAllWindows()
            # if q is pressed, exit
            elif k == ord('q'):
                exit()
        else:
            cv2.waitKey(25)
        ret, frame = cap.read()
        
    
    cap.release()
    cv2.destroyAllWindows()

####################################################################################
#   donat MOT + Video = show BB in Video and allow to go Forward and Back in Time  #
####################################################################################
def displayVideoWithBBoxesFowardBarck(MOT_path, video_path, start_frame=0):
    frameHistory = []
    cap = cv2.VideoCapture(video_path)
    
    # delete 1st line of MOT
    with open(MOT_path, 'r') as fin:
        data = fin.read().splitlines(True)
    
    if data[0].split(',')[0] == 'frame':
        MOT = data[1:]
    else:
        MOT = data       
    MOT = np.array([i.split(',') for i in MOT]).astype(float)
    # get number of frames
    num_frames = np.max(MOT[:,0].astype(int))
    

    # read Mot file from csv
    # S'HA DELIMINAR LA 1A LINEA DEL MOT
    #MOT = np.loadtxt(MOT_path, delimiter=',')
    
    frame_indices = MOT[:,0].astype(int)

    ret, frame = cap.read()
    frame_cnt = 0
    while ret:
        if frame_cnt > num_frames:
            break
        frame_cnt += 1
        print("Frame ", frame_cnt, " of ", int(num_frames), " loaded.")
        mask = frame_indices == frame_cnt
        rows = MOT[mask]

        for row in rows:
            try:
                numFrame, id, x1, y1, w, h, score, x3d, y3d, z3d = row
            except:
                try:
                    numFrame, id, x1, y1, w, h, score = row
                except:
                    numFrame, id, x1, y1, w, h, score, _, _ = row
            numFrame = int(numFrame)
            id = int(id)

            if x1 != -1 and y1 != -1 and w != -1 and h != -1:  
                if score < 0.075:
                    continue              
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x1+w), int(y1+h)), bright_colors[id%len(bright_colors)], 1)
                cv2.putText(frame, str(int(id)), (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color_black, 1)
                #print(id, x1, y1, w, h, score)
                pass

        #frame = frame[1800:, 2000:]
        def drawROIs(frame):
            height, width, _ = frame.shape

            # Dimensiones del cuadrado y solapamiento
            square_size = 640
            overlap_ratio = 0.20
            step_size = int(square_size * (1 - overlap_ratio))

            # Dibujar los cuadrados
            for y in range(0, height - square_size + 1, step_size):
                for x in range(0, width - square_size + 1, step_size):
                    top_left = (x, y)
                    bottom_right = (x + square_size, y + square_size)
                    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Color verde y grosor de línea de 2 píxeles
            return frame

        frame = drawROIs(frame)
        #frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        frameHistory.append(frame)

        ret, frame = cap.read()

    currentFrame = 0
    while True:
        frame2show = frameHistory[currentFrame]
        #frame2show = frame2show[2000:3100, 2000:]
        frame2show = cv2.resize(frame2show, (frame2show.shape[1]//2, frame2show.shape[0]//2))
        cv2.imshow(str(video_path.split("\\")[-1].split(".")[0]), frame2show)
        k = cv2.waitKey(0)

        if k == 32:
            cv2.destroyAllWindows()
        elif k == ord('q'):
            exit()
        elif k == ord('a'):
            currentFrame -= 1
            if currentFrame <= 0:
                currentFrame = 0
            print("frame", str(currentFrame).zfill(4), ".jpg")
        elif k == ord('z'):
            currentFrame -= 50
            if currentFrame <= 0:
                currentFrame = 0
            print("frame", str(currentFrame).zfill(4), ".jpg")
        elif k == ord('d'):
            currentFrame += 1
            if currentFrame >= len(frameHistory)-1:
                currentFrame = len(frameHistory)-1
            print("frame", str(currentFrame).zfill(4), ".jpg")
        elif k == ord('c'):
            currentFrame += 50
            if currentFrame >= len(frameHistory)-1:
                currentFrame = len(frameHistory)-1
            print("frame", str(currentFrame).zfill(4), ".jpg")
        elif k == ord('i'):
            currentFrame = 0
            print("frame", str(currentFrame).zfill(4), ".jpg")
        elif k == ord('o'):
            currentFrame = len(frameHistory)-1
            print("frame", str(currentFrame).zfill(4), ".jpg")

    cap.release()
    cv2.destroyAllWindows()
    


def compareVisuallyMOTs(MOT_n,MOT_s, MOT_m, MOT_l, MOT_x, vid, graph, Layer_1, Layer_2, Layer_3, Layer_4):
    # Layer_1: Eliminar aquelles bboxes que tenen un score < 0.5

    # Elimino la capçalera dels fitxers MOT
    with open(MOT_n, 'r') as fin:
        data = fin.read().splitlines(True)
        if data[0].split(',')[0] == 'frame':
            MOT_n = data[1:]
        else:
            MOT_n = data

    with open(MOT_s, 'r') as fin:
        data = fin.read().splitlines(True)
        if data[0].split(',')[0] == 'frame':
            MOT_s = data[1:]
        else: 
            MOT_s = data
    
    with open(MOT_m, 'r') as fin:
        data = fin.read().splitlines(True)
        if data[0].split(',')[0] == 'frame':
            MOT_m = data[1:]
        else:
            MOT_m = data

    with open(MOT_l, 'r') as fin:
        data = fin.read().splitlines(True)
        if data[0].split(',')[0] == 'frame':
            MOT_l = data[1:]
        else: 
            MOT_l = data
    
    with open(MOT_x, 'r') as fin:
        data = fin.read().splitlines(True)
        if data[0].split(',')[0] == 'frame':
            MOT_x = data[1:]
        else:
            MOT_x = data
    
    # Llegim MOT i ho guardo en una MATRIX
    MOT_n = np.array([i.split(',') for i in MOT_n]).astype(float)
    MOT_s = np.array([i.split(',') for i in MOT_s]).astype(float)
    MOT_m = np.array([i.split(',') for i in MOT_m]).astype(float)
    MOT_l = np.array([i.split(',') for i in MOT_l]).astype(float)
    MOT_x = np.array([i.split(',') for i in MOT_x]).astype(float)

    # Eliminem aquelles linies que no tenen score > 0.5
    if Layer_1:
        mask_n = np.logical_or.reduce([MOT_n[:,6] > 0.5, MOT_n[:,2] == -1, MOT_n[:,3] == -1, MOT_n[:,4] == 0, MOT_n[:,5] == 0])
        mask_s = np.logical_or.reduce([MOT_s[:,6] > 0.5, MOT_s[:,2] == -1, MOT_s[:,3] == -1, MOT_s[:,4] == 0, MOT_s[:,5] == 0])
        mask_m = np.logical_or.reduce([MOT_m[:,6] > 0.5, MOT_m[:,2] == -1, MOT_m[:,3] == -1, MOT_m[:,4] == 0, MOT_m[:,5] == 0])
        mask_l = np.logical_or.reduce([MOT_l[:,6] > 0.5, MOT_l[:,2] == -1, MOT_l[:,3] == -1, MOT_l[:,4] == 0, MOT_l[:,5] == 0])
        mask_x = np.logical_or.reduce([MOT_x[:,6] > 0.5, MOT_x[:,2] == -1, MOT_x[:,3] == -1, MOT_x[:,4] == 0, MOT_x[:,5] == 0])

        MOT_n = MOT_n[mask_n]
        MOT_s = MOT_s[mask_s]
        MOT_m = MOT_m[mask_m]
        MOT_l = MOT_l[mask_l]
        MOT_x = MOT_x[mask_x]

    # Eliminem aquells BBoxes que apareixen pocs frames i son mes grans que la mitjana i que el seu moviment es 0 (static)
    def cleanBBoxes(MOT, title):

        def getData2Plot(MOT):
            ids =  np.unique(MOT[:,1]).astype(int)
            len_ids = []
            for id in ids:
                mask = MOT[:,1] == id
                len_ids.append(len(MOT[mask]))

            area_ids = []
            for id in ids:
                mask = MOT[:,1] == id
                area_ids.append(np.mean(MOT[mask][:,4]*MOT[mask][:,5]))       

            return len_ids, area_ids
           


        # plot scatter len_ids vs area_ids INICIALS
        len_ids, area_ids = getData2Plot(MOT)
        plt.scatter(len_ids, area_ids, s=100)
        print("Initial data", len(len_ids))

        if Layer_2:            
            result = []
            for i in range(len(len_ids)):
                if len_ids[i] > 10 and area_ids[i] < 10000:
                    result.append(True)
                else:
                    result.append(False)
            mask_s = np.array(result)
            ids =  np.unique(MOT[:,1]).astype(int)
            ids = ids[mask_s]
            # eliminem de MOT_s aquells que no compleixen la condicio de area i longitud
            mask = np.array([True if i in ids else False for i in MOT[:,1]])
            MOT = MOT[mask]
         
        # plot scatter len_ids vs area_ids DESPRES ELIMINAR ELS QUE NO CUMPLEIXEN LA 1A CONDICIO
        len_ids, area_ids = getData2Plot(MOT)
        # plot scatter len_ids vs area_ids
        plt.scatter(len_ids, area_ids, s=30, color='yellow')
        print("2a condicio", len(len_ids))


        # Busco i elimino aquells id que tenen un moviment null (static)
        if Layer_3:
            ids =  np.unique(MOT[:,1]).astype(int)
            line2remove = []
            for id in ids:
                mask = MOT[:,1] == id
                if np.var(MOT[mask][:,2][-5:]) == 0 and np.var(MOT[mask][:,3][-5:]) == 0:
                    
                    # Detecto que hi ha alguna bbox que no es mou
                    # Elimino totes les bboxes d'aquest id (del final de la seva existencia) per deixar nomes el moviment
                    # Elimino desde el final fins el primer frame fins que es troba un moviment                
                    for i in range(len(MOT[mask])-1, 0, -1):
                        if np.var(MOT[mask][:,2][i-5:i]) != 0 or np.var(MOT[mask][:,3][i-5:i]) != 0:
                            # elimino desde i-4 fins el final                 
                            last_frame = MOT[mask][:,0][i-3:]
                            line2remove.append([id, last_frame])
                            break   
                                
            # Elimino les linies que no es mouen
            for line in line2remove:
                mask = np.logical_and(MOT[:,1] == line[0], np.logical_and(MOT[:,0] >= line[1][0], MOT[:,0] <= line[1][-1]))
                MOT = MOT[~mask]

                
            
        # plot scatter len_ids vs area_ids DESPRES ELIMINAR ELS QUE NO CUMPLEIXEN LA 2A CONDICIO
        len_ids, area_ids = getData2Plot(MOT)
        plt.scatter(len_ids, area_ids, s=10, color='red')
        print("3a condicio", len(len_ids))


        if Layer_4:            
            result = []
            for i in range(len(len_ids)):
                if len_ids[i] > 10 and area_ids[i] < 10000:
                    result.append(True)
                else:
                    result.append(False)
            mask_s = np.array(result)
            ids =  np.unique(MOT[:,1]).astype(int)
            ids = ids[mask_s]
            # eliminem de MOT_s aquells que no compleixen la condicio de area i longitud
            mask = np.array([True if i in ids else False for i in MOT[:,1]])
            MOT = MOT[mask]

        # plot scatter len_ids vs area_ids DESPRES ELIMINAR ELS QUE NO CUMPLEIXEN LA 1A CONDICIO
        len_ids, area_ids = getData2Plot(MOT)
        # plot scatter len_ids vs area_ids
        plt.scatter(len_ids, area_ids, color='black', marker='x')
        print("3a condicio", len(len_ids))


        plt.xlabel("Number of frames")
        plt.ylabel("Mean area")
        plt.legend(["All data", "Filtered data", "Static data", "Static data filtered"])
        plt.title("Number of frames vs Mean area (" + title + ")")
        
        if graph:
            plt.show()

        return MOT
    
    MOT_n = cleanBBoxes(MOT_n, "Nano")
    MOT_s = cleanBBoxes(MOT_s, "Small")
    MOT_m = cleanBBoxes(MOT_m, "Medium")
    MOT_l = cleanBBoxes(MOT_l, "Large")
    MOT_x = cleanBBoxes(MOT_x, "Extra")



    

    # Llegim el video
    cap = cv2.VideoCapture(vid)
    ret, frame = cap.read()
    frame_cnt = 0

    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    yellow = (0, 255, 255)
    black = (0, 0, 0)

    while ret:
        frame_cnt += 1
        # Selecionem les files del frame actual
        mask_n = MOT_n[:,0] == frame_cnt
        mask_s = MOT_s[:,0] == frame_cnt
        mask_m = MOT_m[:,0] == frame_cnt
        mask_l = MOT_l[:,0] == frame_cnt
        mask_x = MOT_x[:,0] == frame_cnt

        rows_n = MOT_n[mask_n]
        rows_s = MOT_s[mask_s]
        rows_m = MOT_m[mask_m]
        rows_l = MOT_l[mask_l]
        rows_x = MOT_x[mask_x]

        for row in rows_n:
            numFrame, id, x1, y1, w, h, score, x3d, y3d, z3d = row
            numFrame = int(numFrame)
            id = int(id)

            if x1 != -1 and y1 != -1 and w != -1 and h != -1:  
                
                if score < 0.5:
                    continue              
                #cv2.rectangle(frame, (int(x1), int(y1)), (int(x1+w), int(y1+h)), bright_colors[id%len(bright_colors)], 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x1+w), int(y1+h)), black, 2)
                cv2.putText(frame, str(int(id)), (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2)
                cv2.putText(frame, "IOU", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2)
                pass
        
        for row in rows_s:
            numFrame, id, x1, y1, w, h, score, x3d, y3d, z3d = row
            numFrame = int(numFrame)
            id = int(id)

            if x1 != -1 and y1 != -1 and w != -1 and h != -1:  
                
                if score < 0.5:
                    continue              
                #cv2.rectangle(frame, (int(x1), int(y1)), (int(x1+w), int(y1+h)), bright_colors[id%len(bright_colors)], 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x1+w), int(y1+h)), red, 2)
                cv2.putText(frame, str(int(id)), (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)
                cv2.putText(frame, "GIOU", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)
                pass
        
        for row in rows_m:
            numFrame, id, x1, y1, w, h, score, x3d, y3d, z3d = row
            numFrame = int(numFrame)
            id = int(id)

            if x1 != -1 and y1 != -1 and w != -1 and h != -1:  
                if score < 0.5:
                    continue              
                #cv2.rectangle(frame, (int(x1), int(y1), int(x1+w), int(y1+h)), bright_colors[id%len(bright_colors)], 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x1+w), int(y1+h)), blue, 2)
                cv2.putText(frame, str(int(id)), (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, blue, 2)
                cv2.putText(frame, "CIOU", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, blue, 2)
                pass
        
        for row in rows_l:
            numFrame, id, x1, y1, w, h, score, x3d, y3d, z3d = row
            numFrame = int(numFrame)
            id = int(id)

            if x1 != -1 and y1 != -1 and w != -1 and h != -1:  
                if score < 0.5:
                    continue              
                #cv2.rectangle(frame, (int(x1), int(y1), int(x1+w), int(y1+h)), bright_colors[id%len(bright_colors)], 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x1+w), int(y1+h)), green, 2)
                cv2.putText(frame, str(int(id)), (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, green, 2)
                cv2.putText(frame, "DIOU", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, green, 2)
                pass

        for row in rows_x:
            numFrame, id, x1, y1, w, h, score, x3d, y3d, z3d = row
            numFrame = int(numFrame)
            id = int(id)

            if x1 != -1 and y1 != -1 and w != -1 and h != -1:  
                if score < 0.5:
                    continue              
                #cv2.rectangle(frame, (int(x1), int(y1), int(x1+w), int(y1+h)), bright_colors[id%len(bright_colors)], 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x1+w), int(y1+h)), yellow, 2)
                cv2.putText(frame, str(int(id)), (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, yellow, 2)
                cv2.putText(frame, "CT_DIST", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, yellow, 2)
                pass


        frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        cv2.imshow(str(vid.split("\\")[-1].split(".")[0]), frame)
        cv2.waitKey(25)
        
        ret, frame = cap.read()



def validateMOTs(MOT_n, MOT_s, MOT_m, MOT_l, MOT_x, LABELS_VAL):

    # Guardem els resultats de la desviacio tipica normalitzada
    std_n = []
    std_s = []
    std_m = []
    std_l = []
    std_x = []
    # Guardem els resultats de la mean normalitzada
    means_n = []
    means_s = []
    means_m = []
    means_l = []
    means_x = []

    # Guardarem el numero de BBoxes de cada arxiu
    len_n = []
    len_s = []
    len_m = []
    len_l = []
    len_x = []
    len_val = []

    # Guardarem el error entre les BBoxes de cada arxiu i les BBoxes de validacio
    error_n = []
    error_s = []
    error_m = []
    error_l = []
    error_x = []

    # Guardem els valors de IOU per cada solapament al arxiu de validacio
    IOU_n = []
    IOU_s = []
    IOU_m = []
    IOU_l = []
    IOU_x = []
    IOU_val = []

    # Guardem la unio de totes les BBoxes
    union_n = []
    union_s = []
    union_m = []
    union_l = []
    union_x = []
    union_val = []

    # Guardem la area de totes les BBoxes
    area_n = []
    area_s = []
    area_m = []
    area_l = []
    area_x = []
    area_val = []

    # Guardem el numero de interseccions que hi ha per cada file
    numIntersections_n = []
    numIntersections_s = []
    numIntersections_m = []
    numIntersections_l = []
    numIntersections_x = []
    numIntersections_val = []

    # Importo les labels de validacio
    VAL_labels = [] # [Name of the file, data]
    for val in os.listdir(LABELS_VAL):
        with open(os.path.join(LABELS_VAL, val), 'r') as fin:
            data = fin.read().splitlines(True)
            VAL_labels.append([val, data])
            len_val.append(len(data))


        # Guardo en llistes els noms dels fitxers de labels
        
        # read the lines of the file
        with open(os.path.join(MOT_n, val), 'r') as fin:
            data = fin.read().splitlines(True)
            #len_n.append(len_val[-1] - len(data))
            len_n.append(len(data))
            area = 0
            for dat in data:
                _, x0, y0, w, h, _ = dat.split(',')
                x0 = float(x0)
                y0 = float(y0)
                w = float(w)
                h = float(h)
                # Creo box amb shapely de la label de validacio
                box_n = sg.box(x0, y0, x0+w, y0+h)
                area += box_n.area
            area_n.append(area)

        # read the lines of the file
        with open(os.path.join(MOT_s, val), 'r') as fin:
            data = fin.read().splitlines(True)
            #len_s.append(len_val[-1] - len(data))
            len_s.append(len(data))
            area = 0
            for dat in data:
                _, x0, y0, w, h, _ = dat.split(',')
                x0 = float(x0)
                y0 = float(y0)
                w = float(w)
                h = float(h)
                # Creo box amb shapely de la label de validacio
                box_s = sg.box(x0, y0, x0+w, y0+h)
                area += box_s.area
            area_s.append(area)

        # read the lines of the file
        with open(os.path.join(MOT_m, val), 'r') as fin:
            data = fin.read().splitlines(True)
            #len_m.append(len_val[-1] - len(data))
            len_m.append(len(data))
            area = 0
            for dat in data:
                _, x0, y0, w, h, _ = dat.split(',')
                x0 = float(x0)
                y0 = float(y0)
                w = float(w)
                h = float(h)
                # Creo box amb shapely de la label de validacio
                box_m = sg.box(x0, y0, x0+w, y0+h)
                area += box_m.area
            area_m.append(area)

        # read the lines of the file
        with open(os.path.join(MOT_l, val), 'r') as fin:
            data = fin.read().splitlines(True)
            #len_l.append(len_val[-1] - len(data))
            len_l.append(len(data))
            area = 0
            for dat in data:
                _, x0, y0, w, h, _ = dat.split(',')
                x0 = float(x0)
                y0 = float(y0)
                w = float(w)
                h = float(h)
                # Creo box amb shapely de la label de validacio
                box_l = sg.box(x0, y0, x0+w, y0+h)
                area += box_l.area
            area_l.append(area)

        # read the lines of the file
        with open(os.path.join(MOT_x, val), 'r') as fin:
            data = fin.read().splitlines(True)
            #len_x.append(len_val[-1] - len(data))
            len_x.append(len(data))
            area = 0
            for dat in data:
                _, x0, y0, w, h, _ = dat.split(',')
                x0 = float(x0)
                y0 = float(y0)
                w = float(w)
                h = float(h)
                # Creo box amb shapely de la label de validacio
                box_x = sg.box(x0, y0, x0+w, y0+h)
                area += box_x.area
            area_x.append(area)


    

    # Busco el error (distancia entre centres) entre les BBoxes de cada arxiu i les BBoxes de validacio
    ###############################################################################
    def getErrorDist(MOT, fileName, box_val):
        min_error = np.inf   
        #cv2.rectangle(img, (int(x0), int(y0)), (int(x0+w), int(y0+h)), (0, 255, 0), 2)
        #cv2.imshow("img", img)
        listBBoxes = []
        with open(os.path.join(MOT, fileName), 'r') as file:
            data = file.read().splitlines(True)
            for dat in data:
                _, x0, y0, w, h, _ = dat.split(',')
                x0 = float(x0)
                y0 = float(y0)
                w = float(w)
                h = float(h)
                # Creo box amb shapely de la label de validacio
                box_n = sg.box(x0, y0, x0+w, y0+h)
                listBBoxes.append(box_n)

                # Busco distancia entre les dues bboxes
                error = box_n.centroid.distance(box_val.centroid)
                if error < min_error:
                    min_error = error

        union = unary_union(listBBoxes).area

        return min_error, union
    ###############################################################################

    # Comparo si el IOU es mante en les inferencies
    ###############################################################################
    def getErrorIOU(MOT, fileName, IOU):
        min_error = -np.inf   
        #cv2.rectangle(img, (int(x0), int(y0)), (int(x0+w), int(y0+h)), (0, 255, 0), 2)
        #cv2.imshow("img", img)
        
        with open(os.path.join(MOT, fileName), 'r') as file:
            data = file.read().splitlines(True)
            for dat in data:
                _, x0, y0, w, h, _ = dat.split(',')
                x0 = float(x0)
                y0 = float(y0)
                w = float(w)
                h = float(h)
                # Creo box amb shapely de la label de validacio
                box_n = sg.box(x0, y0, x0+w, y0+h)

                for dat_2 in data:
                    _, x0, y0, w, h, _ = dat_2.split(',')
                    x0 = float(x0)
                    y0 = float(y0)
                    w = float(w)
                    h = float(h)
                    # Creo box amb shapely de la label de validacio
                    box_n_2 = sg.box(x0, y0, x0+w, y0+h)

                    # Calculo IOU entre les dues bboxes
                    iou = box_n.intersection(box_n_2).area / box_n.union(box_n_2).area

                    # Busco distancia entre les dues bboxes
                    error = np.abs(iou - IOU)
                    if error < min_error:
                        min_error = error
                   
        
        return min_error
    ###############################################################################

    # Busco el numero de interseccions en un arxiu
    ###############################################################################
    def getNumIntersections(MOT, fileName):
        numIntersections = 0   
        #cv2.rectangle(img, (int(x0), int(y0)), (int(x0+w), int(y0+h)), (0, 255, 0), 2)
        #cv2.imshow("img", img)
        
        with open(os.path.join(MOT, fileName), 'r') as file:
            data = file.read().splitlines(True)
            for dat in data:
                try:
                    _, x0, y0, w, h, _ = dat.split(',')
                except:
                    _, x0, y0, w, h = dat.split(' ')
                x0 = float(x0)
                y0 = float(y0)
                w = float(w)
                h = float(h)
                # Creo box amb shapely de la label de validacio
                box_1 = sg.box(x0, y0, x0+w, y0+h)

                for dat_2 in data:
                    try:
                        _, x0, y0, w, h, _ = dat_2.split(',')
                    except:
                        _, x0, y0, w, h = dat_2.split(' ')
                    x0 = float(x0)
                    y0 = float(y0)
                    w = float(w)
                    h = float(h)
                    # Creo box amb shapely de la label de validacio
                    box_2 = sg.box(x0, y0, x0+w, y0+h)

                    if box_1 == box_2:
                        continue

                    if box_1.intersects(box_2) or box_1.contains(box_2) or box_2.contains(box_1):
                        numIntersections += 1                    
        
        return numIntersections/2
    ###############################################################################

    for lab in VAL_labels:
        fileName = lab[0]
        data = lab[1]
        print(fileName)
        # Llegim la imatge base
        img = cv2.imread(os.path.join(MOT_n, fileName.split(".")[0] + ".jpg")) 

        listBBoxes = []
        area = 0
        for dat in data:
            _, x0, y0, w, h = dat.split(' ')
            x0 = float(x0)*640 
            y0 = float(y0)*640
            w = float(w)*640
            h = float(h)*640

            x0 = x0 - w/2
            y0 = y0 - h/2

            # Creo box amb shapely de la label de validacio
            box_val = sg.box(x0, y0, x0+w, y0+h)                 
            listBBoxes.append(box_val)
            area += box_val.area
            

            # Busco el error (distancia entre centres) entre les BBoxes de cada arxiu i les BBoxes de validacio
            min_error_n, val_union_n = getErrorDist(MOT_n, fileName, box_val)
            min_error_s, val_union_s = getErrorDist(MOT_s, fileName, box_val)
            min_error_m, val_union_m = getErrorDist(MOT_m, fileName, box_val)
            min_error_l, val_union_l = getErrorDist(MOT_l, fileName, box_val)
            min_error_x, val_union_x = getErrorDist(MOT_x, fileName, box_val)

            error_n.append(min_error_n)
            error_s.append(min_error_s)
            error_m.append(min_error_m)
            error_l.append(min_error_l)
            error_x.append(min_error_x)

            try:
                if union_n[-1] != val_union_n:
                    union_n.append(val_union_n)
                if union_s[-1] != val_union_s:
                    union_s.append(val_union_s)
                if union_m[-1] != val_union_m:
                    union_m.append(val_union_m)
                if union_l[-1] != val_union_l:
                    union_l.append(val_union_l)
                if union_x[-1] != val_union_x:
                    union_x.append(val_union_x)
            except:
                union_n.append(val_union_n)
                union_s.append(val_union_s)
                union_m.append(val_union_m)
                union_l.append(val_union_l)
                union_x.append(val_union_x)

        
        # Busco el numero de interseccions en un arxiu
        numIntersections_n.append(getNumIntersections(MOT_n, fileName))
        numIntersections_s.append(getNumIntersections(MOT_s, fileName))
        numIntersections_m.append(getNumIntersections(MOT_m, fileName))
        numIntersections_l.append(getNumIntersections(MOT_l, fileName))
        numIntersections_x.append(getNumIntersections(MOT_x, fileName))
        numIntersections_val.append(getNumIntersections(LABELS_VAL, fileName))
        if numIntersections_val[-1] > 40:
            print("Error numIntersections_n")

        union_val.append(unary_union(listBBoxes).area)
        area_val.append(area)
        pass

            
    plt.subplot(5, 1, 1)
    plt.plot(error_n)
    # draw mean error
    mean_n = np.mean(error_n)
    plt.axhline(y=mean_n, color='r', linestyle='-')
    # draw std error
    variance_n = np.std(error_n)
    plt.axhline(y=variance_n, color='g', linestyle='-')
    # write the variance on the plot
    plt.text(100, 30, "Desv.Stnd: " + str(variance_n), fontsize=10)
    plt.text(100, 25, "Mean : " + str(mean_n), fontsize=10)
    plt.xlabel("BBoxes")
    plt.ylabel("Error")
    plt.title("Error between BBoxes center and validation center")
    #plt.show()

    plt.subplot(5, 1, 2)
    plt.plot(error_s)
    # draw mean error
    mean_s = np.mean(error_s)
    plt.axhline(y=mean_s, color='r', linestyle='-')
    # draw std error
    variance_s = np.std(error_s)
    plt.axhline(y=variance_s, color='g', linestyle='-')
    # write the variance on the plot
    plt.text(5, 30, "Desv.Stnd: " + str(variance_s), fontsize=10)
    plt.text(5, 25, "Mean : " + str(mean_s), fontsize=10)
    plt.xlabel("BBoxes")
    plt.ylabel("Error")
    plt.title("Error between BBoxes center and validation center")
    #plt.show()

    plt.subplot(5, 1, 3)
    plt.plot(error_m)
    # draw mean error
    mean_m = np.mean(error_m)
    plt.axhline(y=mean_m, color='r', linestyle='-')
    # draw std error
    variance_m = np.std(error_m)
    plt.axhline(y=variance_m, color='g', linestyle='-')
    # write the variance on the plot
    plt.text(5, 30, "Desv.Stnd: " + str(variance_m), fontsize=10)
    plt.text(5, 25, "Mean : " + str(mean_m), fontsize=10)
    plt.xlabel("BBoxes")
    plt.ylabel("Error")
    plt.title("Error between BBoxes center and validation center")
    #plt.show()

    plt.subplot(5, 1, 4)
    plt.plot(error_l)
    # draw mean error
    mean_l = np.mean(error_l)
    plt.axhline(y=mean_l, color='r', linestyle='-')
    # draw std error
    variance_l = np.std(error_l)
    plt.axhline(y=variance_l, color='g', linestyle='-')
    # write the variance on the plot
    plt.text(5, 30, "Desv.Stnd: " + str(variance_l), fontsize=10)
    plt.text(5, 25, "Mean : " + str(mean_l), fontsize=10)
    plt.xlabel("BBoxes")
    plt.ylabel("Error")
    plt.title("Error between BBoxes center and validation center")
    #plt.show()

    plt.subplot(5, 1, 5)
    plt.plot(error_x)
    # draw mean error
    mean_x = np.mean(error_x)
    plt.axhline(y=mean_x, color='r', linestyle='-')
    # draw std error
    variance_x = np.std(error_x)
    plt.axhline(y=variance_x, color='g', linestyle='-')
    # write the variance on the plot
    plt.text(5, 30, "Desv.Stnd: " + str(variance_x), fontsize=10)
    plt.text(5, 25, "Mean : " + str(mean_x), fontsize=10)
    plt.xlabel("BBoxes")
    plt.ylabel("Error")
    plt.title("Error between BBoxes center and validation center")
    plt.show()
    variances =  [variance_n, variance_s, variance_m, variance_l, variance_x]
    max_variance = max(variances)
    min_variance = min(variances)
    std_n.append((variance_n - min_variance) / (max_variance - min_variance))
    std_s.append((variance_s - min_variance) / (max_variance - min_variance))
    std_m.append((variance_m - min_variance) / (max_variance - min_variance))
    std_l.append((variance_l - min_variance) / (max_variance - min_variance))
    std_x.append((variance_x - min_variance) / (max_variance - min_variance))
    
    means = [mean_n, mean_s, mean_m, mean_l, mean_x]
    max_mean = max(means)
    min_mean = min(means)
    means_n.append((mean_n - min_mean) / (max_mean - min_mean))
    means_s.append((mean_s - min_mean) / (max_mean - min_mean))
    means_m.append((mean_m - min_mean) / (max_mean - min_mean))
    means_l.append((mean_l - min_mean) / (max_mean - min_mean))
    means_x.append((mean_x - min_mean) / (max_mean - min_mean))




    plt.subplot(1, 5, 1)
    plt.scatter(union_val, union_n, s=100)
    # draw a trend line
    z = np.polyfit(union_val, union_val, 1)
    p = np.poly1d(z)
    # calculate de variance of the residuals
    residuals = union_n - p(union_val)
    variance_n = np.std(residuals)
    # write the variance on the plot
    plt.text(5, 5, "Desv.Stnd: " + str(variance_n), fontsize=10)
    plt.plot(union_val,p(union_val),"r--")
    plt.xlabel("union_val")
    plt.ylabel("union_n")
    plt.title("union_val vs union_n")
    #plt.show()

    plt.subplot(1, 5, 2)
    plt.scatter(union_val, union_s, s=100)
    # draw a trend line
    z = np.polyfit(union_val, union_val, 1)
    p = np.poly1d(z)
    # calculate de variance of the residuals
    residuals = union_s - p(union_val)
    variance_s = np.std(residuals)
    # write the variance on the plot
    plt.text(5, 5, "Desv.Stnd: " + str(variance_s), fontsize=10)
    plt.plot(union_val,p(union_val),"r--")
    plt.xlabel("union_val")
    plt.ylabel("union_s")
    plt.title("union_val vs union_s")
    #plt.show()

    plt.subplot(1, 5, 3)
    plt.scatter(union_val, union_m, s=100)
    # draw a trend line
    z = np.polyfit(union_val, union_val, 1)
    p = np.poly1d(z)
    # calculate de variance of the residuals
    residuals = union_m - p(union_val)
    variance_m = np.std(residuals)
    # write the variance on the plot
    plt.text(5, 5, "Desv.Stnd: " + str(variance_m), fontsize=10)
    plt.plot(union_val,p(union_val),"r--")
    plt.xlabel("union_val")
    plt.ylabel("union_m")
    plt.title("union_val vs union_m")
    #plt.show()

    plt.subplot(1, 5, 4)
    plt.scatter(union_val, union_l, s=100)
    # draw a trend line
    z = np.polyfit(union_val, union_val, 1)
    p = np.poly1d(z)
    # calculate de variance of the residuals
    residuals = union_l - p(union_val)
    variance_l = np.std(residuals)
    # write the variance on the plot
    plt.text(5, 5, "Desv.Stnd: " + str(variance_l), fontsize=10)
    plt.plot(union_val,p(union_val),"r--")
    plt.xlabel("union_val")
    plt.ylabel("union_l")
    plt.title("union_val vs union_l")
    #plt.show()

    plt.subplot(1, 5, 5)
    plt.scatter(union_val, union_x, s=100)
    # draw a trend line
    z = np.polyfit(union_val, union_val, 1)
    p = np.poly1d(z)
    # calculate de variance of the residuals
    residuals = union_x - p(union_val)
    variance_x = np.std(residuals)
    # write the variance on the plot
    plt.text(5, 5, "Variance: " + str(variance_x), fontsize=10)
    plt.plot(union_val,p(union_val),"r--")
    plt.xlabel("union_val")
    plt.ylabel("union_x")
    plt.title("union_val vs union_x")
    plt.show()
    variances = [variance_n, variance_s, variance_m, variance_l, variance_x]
    max_variance = max(variances)
    min_variance = min(variances)
    std_n.append((variance_n - min_variance) / (max_variance - min_variance))
    std_s.append((variance_s - min_variance) / (max_variance - min_variance))
    std_m.append((variance_m - min_variance) / (max_variance - min_variance))
    std_l.append((variance_l - min_variance) / (max_variance - min_variance))
    std_x.append((variance_x - min_variance) / (max_variance - min_variance))
    



    plt.subplot(1, 5, 1)
    plt.scatter(len_val, len_n, s=100)
    # draw a trend line
    z = np.polyfit(len_val, len_val, 1)
    p = np.poly1d(z)
    # calculate de variance of the residuals
    residuals = len_n - p(len_val)
    variance_n = np.std(residuals)
    # write the variance on the plot
    plt.text(5, 5, "Desv.Stnd: " + str(variance_n), fontsize=10)
    plt.plot(len_val,p(len_val),"r--")
    plt.xlabel("len_val")
    plt.ylabel("len_n")
    plt.title("len_val vs len_n")
    #plt.show()

    plt.subplot(1, 5, 2)
    plt.scatter(len_val, len_s, s=100)
    # draw a trend line
    z = np.polyfit(len_val, len_val, 1)
    p = np.poly1d(z)
    # calculate de variance of the residuals
    residuals = len_s - p(len_val)
    variance_s = np.std(residuals)
    # write the variance on the plot
    plt.text(5, 5, "Desv.Stnd: " + str(variance_s), fontsize=10)
    plt.plot(len_val,p(len_val),"r--")
    plt.xlabel("len_val")
    plt.ylabel("len_s")
    plt.title("len_val vs len_s")
    #plt.show()

    plt.subplot(1, 5, 3)
    plt.scatter(len_val, len_m, s=100)
    # draw a trend line
    z = np.polyfit(len_val, len_val, 1)
    p = np.poly1d(z)
    # calculate de variance of the residuals
    residuals = len_m - p(len_val)
    variance_m = np.std(residuals)
    # write the variance on the plot
    plt.text(5, 5, "Desv.Stnd: " + str(variance_m), fontsize=10)
    plt.plot(len_val,p(len_val),"r--")
    plt.xlabel("len_val")
    plt.ylabel("len_m")
    plt.title("len_val vs len_m")
    #plt.show()

    plt.subplot(1, 5, 4)
    plt.scatter(len_val, len_l, s=100)
    # draw a trend line
    z = np.polyfit(len_val, len_val, 1)
    p = np.poly1d(z)
    # calculate de variance of the residuals
    residuals = len_l - p(len_val)
    variance_l = np.std(residuals)
    # write the variance on the plot
    plt.text(5, 5, "Desv.Stnd: " + str(variance_l), fontsize=10)
    plt.plot(len_val,p(len_val),"r--")
    plt.xlabel("len_val")
    plt.ylabel("len_l")
    plt.title("len_val vs len_l")
    #plt.show()

    plt.subplot(1, 5, 5)
    plt.scatter(len_val, len_x, s=100)
    # draw a trend line
    z = np.polyfit(len_val, len_val, 1)
    p = np.poly1d(z)
    # calculate de variance of the residuals
    residuals = len_x - p(len_val)
    variance_x = np.std(residuals)
    # write the variance on the plot
    plt.text(5, 5, "Variance: " + str(variance_x), fontsize=10)
    plt.plot(len_val,p(len_val),"r--")
    plt.xlabel("len_val")
    plt.ylabel("len_x")
    plt.title("len_val vs len_x")
    plt.show()
    variances = [variance_n, variance_s, variance_m, variance_l, variance_x]
    max_variance = max(variances)
    min_variance = min(variances)
    std_n.append((variance_n - min_variance) / (max_variance - min_variance))
    std_s.append((variance_s - min_variance) / (max_variance - min_variance))
    std_m.append((variance_m - min_variance) / (max_variance - min_variance))
    std_l.append((variance_l - min_variance) / (max_variance - min_variance))
    std_x.append((variance_x - min_variance) / (max_variance - min_variance))



    plt.subplot(1, 5, 1)
    plt.scatter(area_val, area_n, s=100)
    # draw a trend line
    z = np.polyfit(area_val, area_val, 1)
    p = np.poly1d(z)
    # calculate de variance of the residuals
    residuals = area_n - p(area_val)
    variance_n = np.std(residuals)
    # write the variance on the plot
    plt.text(5, 5, "Desv.Stnd: " + str(variance_n), fontsize=10)
    plt.plot(area_val,p(area_val),"r--")
    plt.xlabel("area_val")
    plt.ylabel("area_n")
    plt.title("area_val vs area_n")
    #plt.show()

    plt.subplot(1, 5, 2)
    plt.scatter(area_val, area_s, s=100)
    # draw a trend line
    z = np.polyfit(area_val, area_val, 1)
    p = np.poly1d(z)
    # calculate de variance of the residuals
    residuals = area_s - p(area_val)
    variance_s = np.std(residuals)
    # write the variance on the plot
    plt.text(5, 5, "Desv.Stnd: " + str(variance_s), fontsize=10)
    plt.plot(area_val,p(area_val),"r--")
    plt.xlabel("area_val")
    plt.ylabel("area_s")
    plt.title("area_val vs area_s")
    #plt.show()

    plt.subplot(1, 5, 3)
    plt.scatter(area_val, area_m, s=100)
    # draw a trend line
    z = np.polyfit(area_val, area_val, 1)
    p = np.poly1d(z)
    # calculate de variance of the residuals
    residuals = area_m - p(area_val)
    variance_m = np.std(residuals)
    # write the variance on the plot
    plt.text(5, 5, "Desv.Stnd: " + str(variance_m), fontsize=10)
    plt.plot(area_val,p(area_val),"r--")
    plt.xlabel("area_val")
    plt.ylabel("area_m")
    plt.title("area_val vs area_m")
    #plt.show()

    plt.subplot(1, 5, 4)
    plt.scatter(area_val, area_l, s=100)
    # draw a trend line
    z = np.polyfit(area_val, area_val, 1)
    p = np.poly1d(z)
    # calculate de variance of the residuals
    residuals = area_l - p(area_val)
    variance_l = np.std(residuals)
    # write the variance on the plot
    plt.text(5, 5, "Desv.Stnd: " + str(variance_l), fontsize=10)
    plt.plot(area_val,p(area_val),"r--")
    plt.xlabel("area_val")
    plt.ylabel("area_l")
    plt.title("area_val vs area_l")
    #plt.show()

    plt.subplot(1, 5, 5)
    plt.scatter(area_val, area_x, s=100)
    # draw a trend line
    z = np.polyfit(area_val, area_val, 1)
    p = np.poly1d(z)
    # calculate de variance of the residuals
    residuals = area_x - p(area_val)
    variance_x = np.std(residuals)
    # write the variance on the plot
    plt.text(5, 5, "Desv.Stnd: " + str(variance_x), fontsize=10)
    plt.plot(area_val,p(area_val),"r--")
    plt.xlabel("area_val")
    plt.ylabel("area_x")
    plt.title("area_val vs area_x")
    plt.show()
    variances = [variance_n, variance_s, variance_m, variance_l, variance_x]
    max_variance = max(variances)
    min_variance = min(variances)
    std_n.append((variance_n - min_variance) / (max_variance - min_variance))
    std_s.append((variance_s - min_variance) / (max_variance - min_variance))
    std_m.append((variance_m - min_variance) / (max_variance - min_variance))
    std_l.append((variance_l - min_variance) / (max_variance - min_variance))
    std_x.append((variance_x - min_variance) / (max_variance - min_variance))
    


    plt.subplot(1, 5, 1)
    plt.scatter(numIntersections_val, numIntersections_n, s=100)
    # draw a trend line
    z = np.polyfit(numIntersections_val, numIntersections_val, 1)
    p = np.poly1d(z)
    # calculate de variance of the residuals
    residuals = numIntersections_n - p(numIntersections_val)
    variance_n = np.std(residuals)
    # write the variance on the plot
    plt.text(5, 5, "Desv.Stnd: " + str(variance_n), fontsize=10)
    plt.plot(numIntersections_val,p(numIntersections_val),"r--")
    plt.xlabel("numIntersections_val")
    plt.ylabel("numIntersections_n")
    plt.title("numIntersections_val vs numIntersections_n")
    #plt.show()

    plt.subplot(1, 5, 2)
    plt.scatter(numIntersections_val, numIntersections_s, s=100)
    # draw a trend line
    z = np.polyfit(numIntersections_val, numIntersections_val, 1)
    p = np.poly1d(z)
    # calculate de variance of the residuals
    residuals = numIntersections_s - p(numIntersections_val)
    variance_s = np.std(residuals)
    # write the variance on the plot
    plt.text(5, 5, "Desv.Stnd: " + str(variance_s), fontsize=10)
    plt.plot(numIntersections_val,p(numIntersections_val),"r--")
    plt.xlabel("numIntersections_val")
    plt.ylabel("numIntersections_s")
    plt.title("numIntersections_val vs numIntersections_s")
    #plt.show()

    plt.subplot(1, 5, 3)
    plt.scatter(numIntersections_val, numIntersections_m, s=100)
    # draw a trend line
    z = np.polyfit(numIntersections_val, numIntersections_val, 1)
    p = np.poly1d(z)
    # calculate de variance of the residuals
    residuals = numIntersections_m - p(numIntersections_val)
    variance_m = np.std(residuals)
    # write the variance on the plot
    plt.text(5, 5, "Desv.Stnd: " + str(variance_m), fontsize=10)
    plt.plot(numIntersections_val,p(numIntersections_val),"r--")
    plt.xlabel("numIntersections_val")
    plt.ylabel("numIntersections_m")
    plt.title("numIntersections_val vs numIntersections_m")
    #plt.show()

    plt.subplot(1, 5, 4)
    plt.scatter(numIntersections_val, numIntersections_l, s=100)
    # draw a trend line
    z = np.polyfit(numIntersections_val, numIntersections_val, 1)
    p = np.poly1d(z)
    # calculate de variance of the residuals
    residuals = numIntersections_l - p(numIntersections_val)
    variance_l = np.std(residuals)
    # write the variance on the plot
    plt.text(5, 5, "Desv.Stnd: " + str(variance_l), fontsize=10)
    plt.plot(numIntersections_val,p(numIntersections_val),"r--")
    plt.xlabel("numIntersections_val")
    plt.ylabel("numIntersections_l")
    plt.title("numIntersections_val vs numIntersections_l")
    #plt.show()

    plt.subplot(1, 5, 5)
    plt.scatter(numIntersections_val, numIntersections_x, s=100)
    # draw a trend line
    z = np.polyfit(numIntersections_val, numIntersections_val, 1)
    p = np.poly1d(z)
    # calculate de variance of the residuals
    residuals = numIntersections_x - p(numIntersections_val)
    variance_x = np.std(residuals)
    # write the variance on the plot
    plt.text(5, 5, "Desv.Stnd: " + str(variance_x), fontsize=10)
    plt.plot(numIntersections_val,p(numIntersections_val),"r--")
    plt.xlabel("numIntersections_val")
    plt.ylabel("numIntersections_x")
    plt.title("numIntersections_val vs numIntersections_x")
    plt.show()
    variances = [variance_n, variance_s, variance_m, variance_l, variance_x]
    max_variance = max(variances)
    min_variance = min(variances)
    std_n.append((variance_n - min_variance) / (max_variance - min_variance))
    std_s.append((variance_s - min_variance) / (max_variance - min_variance))
    std_m.append((variance_m - min_variance) / (max_variance - min_variance))
    std_l.append((variance_l - min_variance) / (max_variance - min_variance))
    std_x.append((variance_x - min_variance) / (max_variance - min_variance))

    # plot the std
    plt.subplot(1, 2, 1)
    std = np.array([std_n, std_s, std_m, std_l, std_x])
    std = std.T
    experimentName = ['Error center BBoxes', 'union_val vs union_x', 'len_val vs len_x', 'area_val vs area_x', 'numIntersections_val vs numIntersections_x']
    yoloName = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x']
    plt.xticks(np.arange(len(yoloName)), yoloName)
    plt.yticks(np.arange(len(experimentName)), experimentName)
    plt.title("Std")
    plt.imshow(std, cmap='grey', interpolation='nearest')
    plt.colorbar()
    #plt.show()

    # plot the mean
    plt.subplot(1, 2, 2)
    mean = np.array([means_n, means_s, means_m, means_l, means_x])
    mean = mean.T
    experimentName = ['Error center BBoxes']
    yoloName = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x']
    plt.xticks(np.arange(len(yoloName)), yoloName)
    plt.yticks(np.arange(len(experimentName)), experimentName)
    plt.title("Mean")
    plt.imshow(mean, cmap='grey', interpolation='nearest')
    plt.colorbar()
    plt.show()
    pass


if __name__ == "__main__":
    

    iou = "/home/usuaris/imatge/marc.corretge/AntTracking/YOLO_TRACKING/resultsEvalPerMemoria_iouSplit-0.60/ant_subset_1-024_YOLOv8n_Marc_MOT.txt"
    giou = "/home/usuaris/imatge/marc.corretge/AntTracking/YOLO_TRACKING/resultsEvalPerMemoria_giouSplit-0.60/ant_subset_1-024_YOLOv8n_Marc_MOT.txt"
    ciou = "/home/usuaris/imatge/marc.corretge/AntTracking/YOLO_TRACKING/resultsEvalPerMemoria_ciouSplit-0.60/ant_subset_1-024_YOLOv8n_Marc_MOT.txt"
    diou = "/home/usuaris/imatge/marc.corretge/AntTracking/YOLO_TRACKING/resultsEvalPerMemoria_diouSplit-0.60/ant_subset_1-024_YOLOv8n_Marc_MOT.txt"
    ct_dist = "/home/usuaris/imatge/marc.corretge/AntTracking/YOLO_TRACKING/resultsEvalPerMemoria_ct-distSplit-0.60/ant_subset_1-024_YOLOv8n_Marc_MOT.txt"
    
    #compareVisuallyMOTs(MOT_n, MOT_s, MOT_m, MOT_l, MOT_x, "/mnt/work/datasets/AntTracking/videos/ant_subset_1/ant_subset_1-024.mp4", graph=True, Layer_1=True, Layer_2=True, Layer_3=True, Layer_4=True)
    compareVisuallyMOTs(iou, giou, ciou, diou, ct_dist, "/mnt/work/datasets/AntTracking/videos/ant_subset_1/ant_subset_1-024.mp4", graph=False, Layer_1=False, Layer_2=False, Layer_3=False, Layer_4=False)

    #video2Frames("/mnt/work/datasets/AntTracking/videos/ant_subset_1/ant_subset_1-024.mp4", "/mnt/work/datasets/AntTracking/videos/ant_subset_1/frames")
    displayVideoWithBBoxes("/home/usuaris/imatge/marc.corretge/AntTracking/YOLO_TRACKING/resultsEvalPerMemoria_giouSplit-0.60/ant_subset_1-024_YOLOv8n_Marc_MOT.txt", "/mnt/work/datasets/AntTracking/videos/ant_subset_1/ant_subset_1-024.mp4", frameAndWait=True)

    displayVideoWithBBoxesFowardBarck("/home/usuaris/imatge/marc.corretge/AntTracking/YOLO_TRACKING/resultsEvalPerMemoria_giouSplit-0.60/ant_subset_1-024_YOLOv8n_Marc_MOT.txt", "/mnt/work/datasets/AntTracking/videos/ant_subset_1/ant_subset_1-024.mp4") 

    #validateMOTs(iou, giou, ciou, diou, ct_dist, LABELS_VAL)
    
    



    






    ####################################################
    # SCRIPTS PER GENERAR DADES PER LA MEMORIA DEL TFG #
    ####################################################

    predict_path = r"E:\TFG\manualTrackAnnotation\ant_subset_1-024_MOT_OCSORT.txt"
    gt_path = r"E:\TFG\manualTrackAnnotation\GT\ant_subset_1-024_GT.txt"
    # delete 1st line of MOT
    with open(predict_path, 'r') as fin:
        data = fin.read().splitlines(True)
    
    if data[0].split(',')[0] == 'frame':
        MOT = data[1:]
    else:
        MOT = data       
    MOT = np.array([i.split(',') for i in MOT]).astype(float)
    

    # read Mot file from csv
    # S'HA DELIMINAR LA 1A LINEA DEL MOT
    #MOT = np.loadtxt(MOT_path, delimiter=',')
    
    frame_indices = MOT[:,0].astype(int)
    # create a white image of 4000x3000 using cv2 library
    



    # Suposición: 'MOT' y 'frame_indices' están previamente definidos.
    # Colores y demás configuraciones también están definidas.

    frames2show = [3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103]
    frames2show = [i for i in range(3, 750, 5)]
    img = np.ones((3000, 4000, 3), np.uint8) * 255

    # Diccionario para almacenar los centros de los rectángulos por id
    centers = {}

    for frame_cnt in frames2show:

        mask = frame_indices == frame_cnt
        rows = MOT[mask]

        for row in rows:
            try:
                numFrame, id, x1, y1, w, h, score, x3d, y3d, z3d = row
            except ValueError:
                numFrame, id, x1, y1, w, h, score = row
            numFrame = int(numFrame)
            id = int(id)

            if x1 != -1 and y1 != -1 and w != -1 and h != -1:  
                if score < 0.75:
                    continue
                # Dibujar rectángulo
                #if frame_cnt == 3:
                #cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), bright_colors[(id+1) % len(bright_colors)], 6)

                # Calcular el centro del rectángulo
                center_x = int(x1 + w / 2)
                center_y = int(y1 + h / 2)

                # Almacenar el centro en el diccionario
                if id not in centers:
                    centers[id] = []
                centers[id].append((center_x, center_y))

    # Dibujar líneas entre los centros almacenados
    idaa = [2,72]
    #idaa = [11,82,84]
    #idaa = [14,41]
    #idaa = [24,75]
    #idaa = [10]
    for id, points in centers.items():
        for i in range(1, len(points)):
            if id  in idaa: # [24,75], [10]
                cv2.line(img, points[i-1], points[i], bright_colors[(id+1) % len(bright_colors)], 30)
            else:
                cv2.line(img, points[i-1], points[i], bright_colors[(id+1) % len(bright_colors)], 4)

    scale = 4
    img = cv2.resize(img, (4000 // scale, 3000 // scale))
    cv2.imshow("30", img)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

