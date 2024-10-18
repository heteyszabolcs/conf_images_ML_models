# Machine Learning models to classify confocal images based on stem cell colony distribution 

# Random Forest
## Usage

Run _main.py_ function as a Python script (e.g. "python main.py" from Anaconda Prompt). The main.py function has the arguments below: <br>

```
-h, --help            show this help message and exit <br>
  --train_tif_dir TRAIN_TIF_DIR <br>
                        Directory containing TIF files of the training set <br>
  --train_jpg_dir TRAIN_JPG_DIR <br>
                        Destination folder for JPEG files of the training set <br>
  --test_tif_dir TEST_TIF_DIR <br>
                        The folder with the tif images for the predictions <br>
  --test_jpg_dir TEST_JPG_DIR <br>
                        The destination folder for JPEG files to be modelled <br>
  --base_dir [BASE_DIR] <br>
                        Directory of the training set <br>
  --threshold [THRESHOLD] <br>
                        Classification threshold <br>
  --model_name MODEL_NAME <br>
                        Arbitrary name of the Random forest model <br>
```

### Pipeline: 
1.) The pipeline starts with converting the confocal images in tif format to jpg format. <br>
<br>
Notes: <br>
If there aren't new tif images for training then the model will use the available training set. <br>
If there are new tif images for training place them into <br> 
  _train/tif/A:_ if there isn't colony in the field <br>
  _train/tif/B:_ if it is an image with colony/colonies. <br>
<br>
2.) Training Random forest on the training set (=converted jpg images) <br>
3.) Evaluating the training <br>
<br>
Notes: <br>
There will be generated some performance plots. These can be found in the _plot_ folder. <br>
4.) Making predictions on new images using the trained and tuned Random forest model <br>
<br>
Notes: <br>
The new images to be tested must place into the _test/tif_ folder before the run. <br> 
<br>
5.) Selecting the images with potential colony/colonies and place them in the "final" folder. <br>


## Requirements

See _requirements.txt_



