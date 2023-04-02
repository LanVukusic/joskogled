import numpy as np
import cv2


# nalaganje zdravih primerov
PATH_ZDRAVI = "../../data/zdrave.txt"  # pacient,stran,pogled,slika,let_do_raka
# nalaganje rakavih primerov
PATH_RAK = "../../data/rakave.txt"  # pacient,stran,pogled,slika,let_do_raka

# nalaganje zdravih primerov
patients_healthy = []  # patient_id, L_CC, L_MLO, R_CC, R_MLO, years_to_cancer

# format: pacient,stran,pogled,slika,let_do_raka
with open(PATH_RAK, "r") as f:
    lines = f.readlines()
    temp = []
    for i in range(0, len(lines)):
        line_parsed = lines[i].strip().split(",")

        photo_path = line_parsed[3]
        temp.append(photo_path)

        if i % 4 == 3:
            temp.insert(0, line_parsed[0])
            temp.append((line_parsed[4]))
            patients_healthy.append(",".join(temp))
            temp = []


# nalaganje rakavih primerov
patients_cancer = []  # patient_id, L_CC, L_MLO, R_CC, R_MLO, years_to_cancer

# format: pacient,stran,pogled,slika,let_do_raka

with open(PATH_ZDRAVI, "r") as f:
    lines = f.readlines()
    temp = []
    for i in range(0, len(lines)):
        line_parsed = lines[i].strip().split(",")

        photo_path = line_parsed[3]
        temp.append(photo_path)

        if i % 4 == 3:
            temp.insert(0, line_parsed[0])
            temp.append((line_parsed[4]))
            patients_cancer.append(",".join(temp))
            temp = []

podatki = []
# zapis v datoteko
with open("../../data/processed_data.txt", "w") as f:
    f.write("\n".join(patients_healthy + patients_cancer))
