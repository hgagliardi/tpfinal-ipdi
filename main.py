import cv2
import numpy as np

def get_crop_face(face, img):
    x = face[0]
    y = face[1]
    w = face[2]
    h = face[3]
    xF=x+w 
    yF=y+h
    crop_face = img[y:yF, x:xF]
    return crop_face

def get_skin_pixels(img):

    skin_pixels=[]

    for row in img:
        for pixel in row:
            Cr = pixel[1]
            Cb = pixel[2]
            if Cb>=77 and Cb<=127 and Cr>=136 and Cr<=173:
                skin_pixels.append(pixel)

    return np.array(skin_pixels)

# def get_mean_pixel(pixels):
    
#     cant=0
    
#     for pixel in pixels:


def main():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('clasificador/train.xml')
    # Read the input image
    img = cv2.imread('imagenes/1214_01_orig.jpg')
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    face = faces[0]
    crop_img = get_crop_face(face, img)
    crop_imgYCbCr = cv2.cvtColor(crop_img, cv2.COLOR_BGR2YCR_CB)
    crop_imgLab = cv2.cvtColor(crop_img, cv2.COLOR_BGR2Lab)    

    skin_pixels = cv2.cvtColor(get_skin_pixels(crop_imgYCbCr), cv2.COLOR_YCR_CB2BGR)
    skin_pixels = cv2.cvtColor(skin_pixels, cv2.COLOR_BGR2Lab)

    # skin_pixels_mean = get_mean_pixel(skin_pixels)
    print(skin_pixels)
    # Display the output
    # cv2.imshow('img', crop_imgYCbCr)
    # cv2.waitKey(10000)

if __name__ == "__main__":
    main()


# If you want pick rectangle: x = 100, y =200, w = 300, h = 400, you should use code:

# crop_img = img[200:600, 100:300]