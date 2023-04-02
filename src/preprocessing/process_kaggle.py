import pandas as pd

IMG_PATH = "../../data/kaggle_additional/"
CSV_PATH = "../../data/kaggle_data.csv"
CSV_OUT_PATH = "../../data/kaggle_data_processed.csv"

# columns: site_id,patient_id,image_id,laterality,view,age,cancer,biopsy,invasive,BIRADS,implant,density,machine_id,difficult_negative_case


def main():
    df = pd.read_csv(CSV_PATH)
    # keep only the columns we need:
    df = df[["patient_id", "laterality", "view", "cancer", "image_id"]]

    # show all values for laterality and view
    print(df["laterality"].unique())
    print(df["view"].unique())
    print(df["cancer"].unique())

    # temp dict to store the patients and their images
    temp_dict = {}

    # iterate over rows
    for index, row in df.iterrows():
        # get patient id
        patient_id = row["patient_id"]

        # get cancer
        cancer = row["cancer"]

        # if patient is not in the dict, add it
        if patient_id not in temp_dict:
            temp_dict[patient_id] = {
                "cancer": cancer,
            }

        # get laterality
        laterality = row["laterality"]

        # get view
        view = row["view"]

        if view not in ["CC", "MLO"]:
            continue

        # get image id
        image_id = row["image_id"]

        # add image to the dict
        path = "{}{}_{}.png".format(IMG_PATH, patient_id, image_id)
        temp_dict[patient_id][laterality + "_" + view] = path

    # dict to dataframe
    out_df = pd.DataFrame.from_dict(
        temp_dict,
        orient="index",
    )
    print(out_df.head(10))

    # save the output dataframe to csv, without the index
    out_df.to_csv(CSV_OUT_PATH, index=False)


if __name__ == "__main__":
    main()
