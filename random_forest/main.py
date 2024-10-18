import os
import argparse
import pandas as pd
from src.image_conversion import convert_tiff_channel_to_jpg, process_folder_tiffs
from src.data_preparation import prepare_data
from src.image_processing import process_image
from src.model_training import train_random_forest, predict_random_forest
import shutil

train_jpg_folder = "data/train/jpg/A"
train_tif_folder = "data/train/tif/A"

# train folder
base_folder = "data/train/jpg/"
# test folder with new jpges
#test_jpg_folder = "data/test/jpg/"
#test_tif_folder = "data/test/tif/"
#model_name = "best_rf_model"

def run_pipeline(train_tif_folder, train_jpg_folder, test_tif_folder, test_jpg_folder, model_name,
                 base_folder = "data/train/jpg/", threshold=0.3):
    # Step 1: Convert TIFF to JPG
    print("Training set: Converting TIFF files to JPG...")
    process_folder_tiffs(train_tif_folder, train_jpg_folder)
    print("Test set: Converting TIFF files to JPG...")
    process_folder_tiffs(test_tif_folder, test_jpg_folder)

    # Step 2: Prepare data
    print("Preparing data...")
    X, y = prepare_data(base_folder, target_size=(24, 24))

    # Step 3: Train Random Forest
    print("Training Random Forest model...")
    model = train_random_forest(X, y, threshold=threshold, model_name=model_name)

    # Step 4: Make predictions
    print("Making predictions...")
    tests = []
    filenames = []
    for jpg in os.listdir(test_jpg_folder):
        if jpg.lower().endswith(('.jpg', '.jpeg')):
            test = process_image(test_jpg_folder + jpg, target_size=(24, 24), normalize=True)
            filenames.append(jpg)
            tests.append(test)
    preds = predict_random_forest(X = tests, model_path=f"model/{model_name}.joblib", threshold=threshold)
    results = pd.DataFrame({'filename': filenames, 'pred_class': preds[0], 'prob': preds[1]})

    # Step 5: Copying positive predictions to the final folder
    print("Selecting positive predictions...")
    pos_preds = results[results["pred_class"] == 1]["filename"].to_list()

    for pos_pred in pos_preds:
        try:
            shutil.copy(test_jpg_folder + "/" + pos_pred, "final/" + pos_pred)
        except Exception as e:
            error_msg = f"{pos_pred} is already in the destination folder! {str(e)}"
            print(f"{error_msg}")

    print("Done!")
    return pos_preds

#run_pipeline(train_tif_folder = train_tif_folder,
#             train_jpg_folder = train_jpg_folder,
#             test_tif_folder = test_tif_folder,
#             test_jpg_folder = test_jpg_folder,
#             model_name = model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the image classification pipeline")
    parser.add_argument("--train_tif_dir", type=str, required=True,
                        help="Directory containing TIF files of the training set")
    parser.add_argument("--train_jpg_dir", type=str, required=True,
                        help="Destination folder for JPEG files of the training set")
    parser.add_argument("--test_tif_dir", type=str, required=True,
                        help="The folder with the tif images for the predictions")
    parser.add_argument("--test_jpg_dir", type=str, required=True,
                        help="The destination folder for JPEG files to be modelled")
    parser.add_argument("--base_dir", type=str, required=False, default= "data/train/jpg/",
                        nargs='?', const="data/train/jpg/",
                        help="Directory of the training set")
    parser.add_argument("--threshold", type=int, required=False, default=0.3,
                        nargs='?', const=0.3,
                        help="Classification threshold")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Arbitrary name of the Random forest model")
    args = parser.parse_args()

    run_pipeline(train_tif_folder = args.train_tif_dir,
                 train_jpg_folder = args.train_jpg_dir,
                 test_tif_folder = args.test_tif_dir,
                 test_jpg_folder = args.test_jpg_dir,
                 base_folder=args.base_dir,
                 threshold = args.threshold,
                 model_name=args.model_name)

