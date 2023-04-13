import numpy as np


def format_csv(in_path, out_path):
    data = np.genfromtxt(in_path, delimiter=",", dtype=str)
    data = data[1:, [1,2,6]]
    np.savetxt(out_path, data, fmt='%s,%s,%s')

def main():
    in_path = "/mnt/c/py-projects/data/kaggle_data.csv"
    out_path = "/mnt/c/py-projects/data/processeda_data_kaggle.txt"
    format_csv(in_path, out_path)


if __name__ == "__main__":
    main()
