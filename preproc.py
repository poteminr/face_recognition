import cv2
import numpy as np

from keras.models import model_from_json

json_decoder = ''
json_encoder = ''

with open('encoder_mri_json.txt', 'r') as file:
    json_encoder = file.readline()

with open('decoder_mri_json.txt', 'r') as file:
    json_decoder = file.readline()

D = model_from_json(json_decoder)
D.load_weights('decoder_weights_mri.h5')

E = model_from_json(json_encoder)
E.load_weights('encoder_weights_mri.h5')

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


def face_embedding(image):
    """
    :param image: Изображение, открытое через cv2.imread()
    :return: Возвращает array с embedding`ом лиц, если в image лиц >1
             Если лицо одно, то возвращает array с embedding лица
    """
    face_list = crop_faces(image)

    if len(face_list) == 1:
        face = face_list[0]
        res_face = cv2.resize(face, (64, 64))

        res_face = np.expand_dims(res_face, 0)
        res_face = np.expand_dims(res_face, -1)
        res_face = res_face / 255.
        face_encoded = E.predict(res_face)
        return np.array(face_encoded)

    elif len(face_list) > 1:
        faces_encoded = []
        for face in face_list:
            res_face = cv2.resize(face, (64, 64))

            res_face = np.expand_dims(res_face, 0)
            res_face = np.expand_dims(res_face, -1)
            res_face = res_face / 255.
            face_vec = E.predict(res_face)

            faces_encoded.append(np.array(face_vec))
        return np.array(faces_encoded)


class face_values_error(Exception):

    pass


def face_distance(image1,image2):
    """

    :param image1: Изображение с одним лицом.
    :param image2: Изображение с одним лицом, если их несколько, падает Error
    :return: Вернет дистанцию между embedding`ами
    """
    image1_vec = face_embedding(image1)
    image2_vec = face_embedding(image2)
    if len(image1_vec) > 1 or len(image2_vec) >1:
        raise face_values_error('Face in image > 1')
    else:
        return distance(image1_vec,image2_vec)




def valid_embedding(image,model):
    """

    :param image: Изображение через cv2.imread
    :param model: Обученный энкодер для валидации
    :return: Вернет array с вектором,если лицо одно.
            Вернет array с array`ями вектором, если больше одного.
    """
    face_list = crop_faces(image)

    if len(face_list) == 1:
        face = face_list[0]
        res_face = cv2.resize(face, (64, 64))

        res_face = np.expand_dims(res_face, 0)
        res_face = np.expand_dims(res_face, -1)
        res_face = res_face / 255.
        face_encoded = model.predict(res_face)
        return np.array(face_encoded)

    elif len(face_list) > 1:
        faces_encoded = []
        for face in face_list:
            res_face = cv2.resize(face, (64, 64))

            res_face = np.expand_dims(res_face, 0)
            res_face = np.expand_dims(res_face, -1)
            res_face = res_face / 255.
            face_vec = model.predict(res_face)

            faces_encoded.append(np.array(face_vec))
        return np.array(faces_encoded)