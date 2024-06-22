# AntTracking

## Entrenament YOLOv8
### YOLOv8_train.py
Abans d'entrenar, haurem de configurar alguns paràmetres del entrenament com són els següents:
* **YOLO_LETTER:** Seleccionem quin model de YOLOv8 volem entrenar ![Model YOLOv8](resources\yolo-comparison-plots.png) El que millors resultats ens ha donat en el nostre entrenament, ha sigut el model **n**
* **YAML_PATH:** Aquest directori ha de contenir l'arxiu **config.yaml** que ha de contenir les dades d'on es troba el dataset que es farà servir per entrenament. Un exemple d'aquest arxiu seria:
```yaml
path: /mnt/work/users/marc.corretge/train_YOLO_imatgesMeves/train/
train: images/train
val: images/val
test: images/test

# Classes
names:
0: ant
```
* **IMGSZ:** Aquesta variable indicarà la mida de les nostres imatges utilitzades per entrenar, en el nostre cas fem servir 640
* **EPOCHS:** Número de epochs que volem duri l'entrenament. Entre 80-100 ja és suficient
* **BATCH:** Indica el nombre d'imatges es processen conjuntament durant una iteració de l'entrenament
* **OUT_NAME:** Nom del model entrenat que volem que es guardi un cop acabat l'entrenament.

Un cop acabat l'entrenament, amb la instrucció **model.metrics** imprimirem per pantalla el resultat de la validació. A més, se'ns generarà un arxiu de log amb tots els resultats i evolució de l'entrenament.

## Inferència YOLOv8
### YOLOv8_detect.py
Si volem fer una prova de detecció sobre una imatge, conjunt d'imatges o vídeo, executant aquesta funció, farem una inferència de prova per poder comprovar que funciona. Aquesta ens mostrarà per pantalla de forma visual el resultat de la inferència, sense generar-nos cap arxiu de detecció.
Accepta tant imatges, vídeos o conjunts d'imatges. Les variables que haurem de modificar són les següents:
* **DIR:** EN aquesta variable posarem el path del que vulguem fer inferència, sigui una carpeta o un arxiu d'imatge/vídeo amb els formats compatibles especificats a les variables *img_extensions* i *vid_extensions*
* **model:** Model de YOLOv8 que vulguem fer per inferir.

El resultat serà una representació gràfica de les deteccions sobre la imatge/vídeo seleccionat.

### run_YOLOv8_inference.py
Si volem obtenir l'arxiu de deteccions d'un vídeo, executarem aquest programa.
Les variables que haurem de modificar són les següents:
* **ROI:** Aquesta variable contindrà la mida de les imatges que es passaran pel model per realitzar la inferència. Aquest valor idealment hauria de coincidir amb el valor de IMGSZ del model
* **OVERLAP:** Valor de superposició entre els retalls de mida ROI [0,1)
* **DETECT/TRACK:** S'ha habilitat l'opció de realitzar tracking o detecció, tots dos casos fent servir únicament YOLOv8. Com a recomanació fer servir només el mode de DETECT, ja que el tracking no està optimitzat ni implementat del tot, perquè farem servir OC-SORT per tracking. Per tant, recomanem DETECT = True i Track = False
* **model:** Introduirem el path d'on es trobi el millor model entrenat.
* **video_path:** Path del vídeo que volem inferir i generar un arxiu de deteccions MOT a partir d'aquest vídeo
* **newFolder:** Introduirem el path on volem que es guardi l'arxiu de deteccions finals MOT

Un cop realitzat la inferència, se'ns generarà un arxiu de deteccions MOT (amb el mateix nom que el vídeo + model del YOLOv8 + nom del dataset amb el qual s'ha entrenat el model que estàs fent servir) . txt o .csv

Per cridar aquest programa, s'esperaran 2 paràmetres, la lletra del model de YOLOv8 que volem fer servir, i el dataset amb el qual s'ha entrenat el model que volem fer servir. Aquests paràmetres es poden eliminar, per la realització del treball van ser necessaris, ja que es feien servir diferents models i datasets.

### run_YOLOv8_inference_640x640_crop.py
Aquest programa el farem servir només com a test, el procés és el mateix que realitza *run_YOLOv8_inference.py* però sobre retalls de 640x640 i sense generar-nos un arxiu de deteccions. Tot i que seria fàcilment aplicable, s'ha fet pensant en ser una eina de comprovació visual. Això en permet comprovar per exemple que la detecció en zones complicades és correcta, sense haver de processar un vídeo sencer.

Les variables són les mateixes que *run_YOLOv8_inference.py* afegint-ne 2:
* **xCrop:** Aquí posarem la coordenada X que volem que es generi el retall respecte al fotograma inicial. Xf = X0 + ROI
* **yCrop:** Aquí posarem la coordenada Y que volem que es generi el retall respecte al fotograma inicial. Yf = Y0 + ROI

Per cridar aquest programa, s'esperaran 2 paràmetres, la lletra del model de YOLOv8 que volem fer servir, i el dataset amb el qual s'ha entrenat el model que volem fer servir. Aquests paràmetres es poden eliminar, per la realització del treball van ser necessaris, ja que es feien servir diferents models i datasets.


## Tracking OC-SORT
Un cop disposem de l'arxiu de deteccions MOT, passarem a realitzar el tracking amb l'algoritme OC-SORT.
Primer de tot haurem de descarregar-nos el repositori
```bash
git clone https://github.com/noahcao/OC_SORT.git
pip install -r requirements.txt
```
Un cop descarregat, és important actualitzar la línia de codi de *run_OC-SORT_tracking.py* on fem el (from ... import *) de tal manera que estigui vinculat correctament.

Les variables que es fan servir en aquest programa són les següents:
* **VIDEO_PATH:** Posarem el path del vídeo que estem inferint. Això simplement el que farà és obtenir la mida del fotograma, un cop ja s'obtingui això ja no es farà servir més.
* **GENERATE_MOT_FILE:** Aquesta variable ha d'estar a True si volem que ens torni l'arxiu de deteccions amb les identitats obtingudes del tracking.
* **Paràmetres del OC-SORT:**
La majoria de paràmetres del OC-SORT estan configurats per defecte, la modificació d'alguns d'ells ens pot ser interessant per tal de millorar els resultats, d'altres no s'ha vist millora en modificar-los. En aquest treball s'ha modificat el paràmetre *asso_func* i s'ha notat una millora important en passar de *iou* a *giou*. Seria interessant provar amb altres paràmetres per si es pogués trobar una millora en els resultats.
* **MOT_FILE:** Path del arxiu de deteccions MOT que volem fer servir per realitzar el tracking
* **direcoty_out_MOT_OCSORT:** Path on volem que es guardi l'arxiu de deteccions MOT amb les identitats obtingudes del tracking

Un cop finalitzat el tracking, se'ns generarà un arxiu de deteccions MOT amb la nova assignació d'identitats.

## Validació detecció
Per la validació de la detecció, disposem de diferents programes per executar:
* **metrics\visual_TP-FP-FN.py** Aquest programa ens permetrà la visualització dels TP, FP i FN a temps real sobre el vídeo/fotogrames que hem fet servir per la inferència. És important que fem servir un arxiu de deteccions i un GT que corresponguin al mateix vídeo.
Les variables que hem de proporcionar són, directori del GT i de l'arxiu de deteccions MOT, el directori dels fotogrames del vídeo i el llindar d'iou_threshold per considerar una detecció com a TP.
* **metrics\mAP.py** Aquest programa ens calcularà la gràfica precision/recall amb diferents valors de IoU threshold, a més també ens proporcionarà els valors de AP per diferent IoU, per tant ens serà fàcil el càlcul de mAP@50, mAP@95 i mAP@50-95.
* **metrics\PR.py** Aquest programa permet diverses representacions diferents que seleccionarem amb la variable option. Necessitarem proporcionar el path d'un GT i un arxiu de deteccions MOT.
A la variable *iou_thresholds* en el 3r camp haurem d'especificar amb quina resolució volen que es calculin les mètriques.
Les opcions de representació són les següents:
* **P vs IoU and R vs IoU:** Precision vs IoU Threshold
* **P-R Curve:** Precision-Recall Curve
* **P-R-F1 vs IoU:** Precision, Recall, and F1-score vs IoU Threshold



## Validació tracking
Per la validació del tracking disposem d'un programa que ens calcularà totes les mètriques de CLEARMOT, IDS i HOTA.
* **metrics\tracking_metrics.ipynb** En l 2a cel·la del notebook, haurem d'especificar el path amb l'arxiu resultant del Tracking, a més del GT. Els resultats ens apareixeran a les 3 últimes cel·les del notebook.

## Útils
L'arxiu **utils.py** conte diferents funcions que durant la realització d'aquest projecte han estat útils per la realització de certes tasques, i que he considerat necessàries/imprescindibles. Per aquest motiu s'han agrupat en aquest arxiu per tal de fer-les accessibles en tot moment.
Algunes d'aquestes funcions es fan servir en el projecte final, altres només s'han fet servir per la realització d'alguns tests, però totes elles han estat de gran ajuda en algun moment del projecte. A continuació es detallen les funcions que es poden trobar en aquest arxiu:

### video2Frames
A partir del path d'un vídeo i del path on volem que es guardin els fotogrames, aquest programa ens permetrà extreure els fotogrames d'un vídeo

### cut_video
Aquesta funció ens permet retallar temporalment un vídeo a partir del segon inicial fins al segon final que desitgem, i exportar-lo com a nou vídeo. Això és especialment útil per tal de reduir el temps d'inferència en cas que només ens interessi treballar sobre una part del vídeo.
En cas que vulguem retallar en lloc d'amb temps, amb fotogrames, haurem de fer aquesta petita modificació a la funció:
```python
init_frame = start_time
end_frame = end_time
```

### process_bboxes
**(Funció de test)** Aquesta funció s'ha fet servir en les funcions de test que es van implementar al principi, tot i no ser útil en el projecte final, l'he mantingut per si en algun moment fos necessària.
A partir de la classe *frame* que retorna el SAHI o YOLO una vegada inferit, aquesta classe conte els bbox de les deteccions, pel que ens permetrà anotar als fotogrames les bbox i els labels corresponents.

### print_bboxes
A partir d'una imatge de fotograma i una llista totes les deteccions d'aquell fotograma, retornem la imatge amb les deteccions anotades a la imatge a més del score. El paràmetre threshold ens permetrà filtrar les deteccions amb un score inferior al threshold.

### compareVid
Aquesta funció ens permetrà visualitzar 2 vídeos al mateix temps per tal de comparar-los. Passem el path dels 2 vídeos, i aquesta funció ens generarà un vídeo amb els 2 vídeos al mateix temps.
Especialment útil per comparar 2 vídeos d'alguna regió problemàtica amb 2 configuracions diferents i així poder comparar-los visualment.

### plot_path
**(Funció de test)** Aquesta funció permetia representar el camí seguit per les formigues. Tot i no ser útil en el projecte final, l'he mantingut per si en algun moment fos necessària.

### list2matrix
Donada una llista de deteccions, aquesta ens ho retorna convertit en una matriu numpy.

### displayVideoWithBBoxes
Aquesta funció ens permet visualitzar en un vídeo les deteccions inferides sobre ell. D'aquesta manera poder comprovar a temps real si les deteccions són correctes o no, o si hi ha algun canvi d'identitat.
Aquesta funció és especialment útil per comprovar de forma ràpida un arxiu de detecció o tracking. Activant el paràmetre frameAndWait podem fer que el vídeo s'aturi en cada fotograma per tal de poder comprovar bé les deteccions, i fins que no premem l'espai, no continuarà.

### displayVideoWithBBoxesFowardBarck
Igual que l'anterior, però aquest cas ens permetrà moure'ns lliurement pels fotogrames. Si no especifiquem cap paràmetre a start_frame, començarem per l'inici del vídeo. Existeixen unes tecles que ens permeten moure'ns lliurement pel vídeo, que són les següents:
* Amb la tecla **a** anirem cap enrere fotograma a fotograma.
* Amb la tecla **d** anirem cap endavant fotograma a fotograma.
* Amb la tecla **z** anirem cap enrere 50 fotogrames.
* Amb la tecla **c** anirem cap endavant 50 fotogrames.
* Amb la tecla **i** ens permetrà anar al principi del vídeo.
* Amb la tecla **o** a l'últim fotograma.
* Amb la tecla **q** sortirem del vídeo.

### compareVisuallyMOTs
Una altra funció que serà de gran ajuda per comparar fins a 5 arxius de detecció/tracking alhora sobre un vídeo, i permetre'ns poder veure amb temps real si algun d'ells té més problemes, errors en detecció o pèrdues d'identitat. Això és especialment útil quan comparem diferents models de YOLO, ja que ens permet veure visualment i quin model té més falses deteccions i quins no fallen tant.
A més a més, incorpora fins a 5 filtres (que podem veure com s'apliquen activant el paràmetre graph = True) i que eliminaran de forma automàtica aquells bboxes erronis, ja siguin perquè tenen una mitja de mida més gran del normal, o perquè per exemple que aparegui un nombre de fotogrames inferior a l'esperat.
Aquests 5 filtres es poden activar via paràmetre, i la visualització gràfica de les deteccions descartades ens permet visualitzar la quantitat de deteccions que estem descartant amb cada filtre.

### validateMOTs
Aquesta funció ens permetia fer una comparativa entre diversos aspectes, comparant sempre la predicció amb el GT. Algunes de les comparatives que ens proporciona son les següents:
* Error de centratge entre la detecció i el GT
* Comparativa del número d'interseccions del GT i de les deteccions
* Àrea d'interseccions entre tots els bboxes en GT i en deteccions
* Àrea total de tots els bbox de deteccions i GT
* Número total de bboxes en deteccions i en GT

En la versió final del projecte, aquesta funció no s'ha fet servir, però en el desenvolupament del projecte ha estat utilitzada per exemple, per comparar el rendiment entre diversos models del YOLOv8.