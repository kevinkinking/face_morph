import cv2
import imageio
from io import BytesIO
import config
  
def create_gif(image_list):  
    frames = []  
    for image in image_list:  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)
    imageio.mimsave('gif.gif', frames, 'GIF', duration = config.gif_speed)

def create_gif_byte(image_list):  
    frames = []  
    for image in image_list:  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)
    gif_stream = BytesIO()
    imageio.mimsave(gif_stream, frames, 'GIF', duration = config.gif_speed)
    return gif_stream.getvalue()

def corp_imgs_scale(imgs, scale = 0.8):
    if scale > 1:
        scale = 1.0
    if scale < 0.1:
        scale = 0.1
    corped_imgs = []
    for img in imgs:
        x_begin = int(img.shape[1] * (1 - scale) / 2)
        y_begin = int(img.shape[0] * (1 - scale) / 2)
        x_end = img.shape[1] - x_begin
        y_end = img.shape[0] - y_begin
        corped_img = img[y_begin:y_end, x_begin:x_end]
        corped_imgs.append(corped_img)
    return corped_imgs

def create_imgs_byte(imgs):
    imgs_encoded = [bytearray(cv2.imencode(config.img_format, img)[1]) for img in imgs]
    return imgs_encoded