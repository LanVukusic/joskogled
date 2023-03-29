import cv2
import glob
from tqdm import tqdm
import numpy as np

# load images from directory data/rakave and data/zdrave
# rescale them and save them to data/processed_data
# without saving them to a list
RES0 = 3152#1576
RES1 = 4096#2048

def process_in_dir(in_path, out_path, k):
    counter = 0
    for img in tqdm(sorted(glob.glob(in_path + "/*.png"))):
        image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        print(image.shape)
        if image.shape[0] == 4096:
            if "_L_" in img:
                image = image[:, :3152]
            else:
                image = image[:, -3152:]
        image = cv2.resize(image, (RES0 // k, RES1 // k), interpolation=cv2.INTER_AREA)
        # 8 bit unsigned integer (0 to 255)
        image = (image / 256).astype(np.uint8)
        print(cv2.imwrite(out_path + "/" + img.split("/")[-1], image))
        counter +=1
        if counter == 5*4:
            break


def load_images():
    k = 8
    process_in_dir("../data/rakave", "../data/processed_data_halfk_small", k)
    process_in_dir("../data/zdrave", "../data/processed_data_halfk_small", k)


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
