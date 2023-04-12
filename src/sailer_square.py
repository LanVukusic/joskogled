import cv2
import glob
from tqdm import tqdm
import numpy as np

# load images from directory data/rakave and data/zdrave
# rescale them and save them to data/processed_data
# without saving them to a list
RES = 512

def process_in_dir(in_path, out_path):
    for img in tqdm(sorted(glob.glob(in_path + "/*.png"))):
        image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        #print(image.shape)
        pad = image.shape[0] - image.shape[1]
        if "_L_" in img:
            image = cv2.copyMakeBorder(image, 0, 0, 0, pad, cv2.BORDER_CONSTANT)
        else:
            image = cv2.copyMakeBorder(image, 0, 0, pad, 0, cv2.BORDER_CONSTANT)
        image = cv2.resize(image, (RES, RES), interpolation=cv2.INTER_AREA)
        # 8 bit unsigned integer (0 to 255)
        #image = (image / 256).astype(np.uint8)
        res = cv2.imwrite(out_path + "/" + img.split("/")[-1], image)
        #print(res)


def load_images():
    process_in_dir("../data/processed_data", "../data/processed_data_512")


if __name__ == "__main__":
    load_images()
# pinky:src marko$ python3 dataloader.py
# getdata monaa
# getdata monaa

# pinky:src marko$ python3 main.py
# mogoce dela
# getdata monaa
# tensor(['00000000-0000-0000-0000-000000000000'], dtype=torch.str)
# Epoch [1/10], Step [10/10], Loss: 0.0000

# pinky:src marko$ python3 main.py
# mogoce dela
# getdata monaa
# tensor(['00000000-0000-0000-0000-000000000000'], dtype=torch.str)
# Epoch [1/10], Step [10/10], Loss: 0.0000


#
