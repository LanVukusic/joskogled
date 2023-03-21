import cv2
import glob
from tqdm import tqdm
import numpy as np

# load images from directory data/rakave and data/zdrave
# rescale them and save them to data/processed_data
# without saving them to a list
def load_images():
    for img in tqdm(glob.glob("../data/rakave/*.png")):
        print()
        image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (1664, 2048), interpolation=cv2.INTER_AREA)
        # 8 bit unsigned integer (0 to 255)
        image = (image / 256).astype(np.uint8)
        cv2.imwrite("../data/processed_data/" + img.split("/")[-1], image)

    for img in tqdm(glob.glob("../data/zdrave/*.png")):
        image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (1664, 2048), interpolation=cv2.INTER_AREA)
        # 8 bit unsigned integer (0 to 255)
        image = (image / 256).astype(np.uint8)
        cv2.imwrite("../data/processed_data/" + img.split("/")[-1], image)


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
