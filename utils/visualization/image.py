import cv2


def draw_rect_on_image(img, rect, color=(0, 0, 255), thickness=2):
    """
    Draw a rectangle on an image
    :param img: the image to draw on
    :param rect: the rectangle to draw
    :param color: the color of the rectangle
    :param thickness: the thickness of the rectangle
    :return: the image with the rectangle drawn on it
    """
    x, y, w, h = rect
    x2 = x + w
    y2 = y + h
    cv2.rectangle(img, (x, y), (x2, y2), color, thickness)
    return img

def draw_dot_on_image(img, dot, r, color=(0, 0, 255), thickness=2):
    """
    Draw a dot on an image
    :param img: the image to draw on
    :param dot: the dot to draw
    :param r: the radius of the dot
    :param color: the color of the dot
    :param thickness: the thickness of the dot
    :return: the image with the dot drawn on it
    """
    x, y = dot
    cv2.circle(img, (x, y), r, color, thickness)
    return img

def draw_text_on_image(img, text, pos, color=(0, 0, 255), thickness=2):
    """
    Draw text on an image
    :param img: the image to draw on
    :param text: the text to draw
    :param pos: the position of the text
    :param color: the color of the text
    :param thickness: the thickness of the text
    :return: the image with the text drawn on it
    """
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)
    return img

