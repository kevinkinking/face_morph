
import imageio
import cv2
from io import BytesIO
  
def create_gif(image_list, gif_name):  
  
    frames = []  
    for image_name in image_list:  
        img = imageio.imread(image_name)
        frames.append(img)  
        print img.shape
    # Save them as frames into a gif  
    f=BytesIO()
    imageio.mimwrite(f, frames, 'GIF', duration = 0.1)  
    return f

def create_gif_mem(image_list, gif_name):  
  
    frames = []  
    for image in image_list:  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)
    # Save them as frames into a gif   
    imageio.mimsave(gif_name, frames, 'GIF', duration = 0.1)  
  
    return  
  
def main():  
    image_list = ['result/a.jpg', 'result/c.jpg', 'result/e.jpg',   
                  'result/g.jpg', 'result/i.jpg']  
    gif_name = 'created_gif.gif'  
    f =  create_gif(image_list, gif_name)
    print f.getvalue()
  
if __name__ == "__main__":  
    main()  