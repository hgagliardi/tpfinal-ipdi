import os
import json
import cv2
from cv2 import data
import numpy as np

def get_crop_face(face, img):
    x = face[0]
    y = face[1]
    w = face[2]
    h = face[3]
    
    xF=x+w - 10
    yF=y+h - 10
    x += 10
    y += 10
    crop_face = img[y:yF, x:xF]
    return crop_face

def get_skin_pixels(img, imagen_original):

    skin_pixels=[]
    pixelPromediado = np.array([0,0,0])
    count=0
    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            Cr = pixel[1]
            Cb = pixel[2]
            if Cb>=77 and Cb<=127 and Cr>=136 and Cr<=173:
                skin_pixels.append(pixel)
                pixelBGR = imagen_original[i,j]
                pixelPromediado += pixelBGR
                count+=1

    pixelPromediado = pixelPromediado/count

    return np.array(skin_pixels), pixelPromediado

def getITA(img_path, face_cascade):
    img = cv2.imread(img_path)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces)>0:
        face = faces[0]
        crop_img = get_crop_face(face, img)
        crop_imgYCbCr = cv2.cvtColor(crop_img, cv2.COLOR_BGR2YCR_CB)
        # cv2.imshow('img', crop_img)
        # cv2.waitKey(3000)

        skin_pixels_YCR_CB, pixelPromediado = get_skin_pixels(crop_imgYCbCr, crop_img)
        
        im = np.array((pixelPromediado[0],pixelPromediado[1],pixelPromediado[2]),np.uint8).reshape(1,1,3)
        # print('pixel promediado me dio: ', pixelPromediado)
        pixelPromediadoConvertido = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        L = pixelPromediadoConvertido[0,0,0]
        a = pixelPromediadoConvertido[0,0,1]
        b = pixelPromediadoConvertido[0,0,2]

        # print('cielab me dio: ', L,', ', a,', ', b)
        ITA = (np.arctan((L-50)/b) * 180 )/ np.pi

        print('Imprimo el ITA ', ITA)
        return ITA
        # print(skin_pixels)
        # Display the output
        # cv2.imshow('img', skin_pixels_YCR_CB)
        # cv2.waitKey(10000)

def getITARange(ITAValue):
    if ITAValue>50:
        return 1
    elif ITAValue >=25:
        return 2
    elif ITAValue >=0:
        return 3
    elif ITAValue >=-25:
        return 4
    elif ITAValue >=-50:
        return 5
    else:
        return 6

def main():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('clasificador/train.xml')
    json_file = open('imagenes/CasualConversations.json')
    data_json = json.load(json_file)

    for key in data_json:
        e=data_json[key]
        dark_files=e['dark_files']
        files=e['files']
        label=e['label']

        fotos_oscuras=[]
        prom_ITA_dark=0
        count_dark=0
        print("FOTOS OSCURAS")
        if len(dark_files) > 0:
            for dark_elem in dark_files:
                dark_name=dark_elem[:-4]+'_orig.jpg'
                imgPath='imagenes/'+dark_name
                fotos_oscuras.append(imgPath)
                if os.path.isfile(imgPath):
                    ITA = getITA(imgPath, face_cascade)
                    prom_ITA_dark+=ITA
                    count_dark+=1
            prom_ITA_dark=prom_ITA_dark/count_dark
            print(prom_ITA_dark)
            print("Le pegué?")
            print(getITARange(prom_ITA_dark))
            print(label['skin-type'])



        prom_ITA_comun=0
        count_comun=0
        print("FOTOS COMUNES")
        if len(files) > 0:
            for file in files:
                file_name=file[:-4]+'_orig.jpg'
                imgPath='imagenes/'+file_name
                if os.path.isfile(imgPath) and not imgPath in fotos_oscuras:
                    ITA = getITA(imgPath, face_cascade)
                    prom_ITA_comun+=ITA
                    count_comun+=1
            prom_ITA_comun=prom_ITA_comun/count_comun
            print(prom_ITA_comun)

        exit()



    exit()
    # # Iterating through the json
    # # list
    # for i in data['emp_details']:
    #     print(i)
    
    # # Closing file
    # f.close()

    imgDir='imagenes/'
    obj = os.scandir(imgDir)
 
    # Listo todos los directorios y archivos
    for folder in obj :
        imgDir='imagenes/'
        if folder.is_dir():
            for subfolder in os.scandir(folder):
                imgDirFolder=imgDir+folder.name+"/"
                if subfolder.is_dir():
                    imgDirFolder+=subfolder.name

                    for img in os.scandir(subfolder):
                        if img.is_file():
                            imgPath=imgDirFolder+"/"+img.name
                            print("Imagen: " + imgPath)
                            getITA(imgPath, face_cascade)
    
    exit()

  


if __name__ == "__main__":
    main()


# If you want pick rectangle: x = 100, y =200, w = 300, h = 400, you should use code:

# crop_img = img[200:600, 100:300]