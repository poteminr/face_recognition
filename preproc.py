import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def crop_faces(image):
    """

    :param image: Картинка (cv2.imread())
    :return: list со всеми кропнутыми лицами(даже если лицо одно)
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # конвертируем изображение в RGB
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # делаем изображение ЧБ
    faces = face_cascade.detectMultiScale(gray_image, 1.25, 6)  # находим лица
    image_copy = np.copy(image)

    face_crop = []
    for f in faces:
        x, y, w, h = [v for v in f]
        #  cv2.rectangle(image_copy, (x,y), (x+w, y+h), (255,0,0), 3)
        face_crop.append(gray_image[y:y + h, x:x + w])

    return face_crop


def rotate_image(image, angle):
    """

    :param image: Картинка (cv2.imread())
    :param angle: Угол поворота картинки в градусах
    :return: Повернутая картинка на заданный угол
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def distance(emb1, emb2):
    """

    :param emb1: Первый вектор признаков
    :param emb2: Второй вектор признаков
    :return: Расстояние между векторами
    """
    return np.sum(np.square(emb1 - emb2))