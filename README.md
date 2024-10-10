# Machine Learning models to classify confocal images based on stem cell colony distribution 

## Usage

Example on the Random forest model: 

module _unstack.py_: 

Convert tif files coming from confocal microscopes containing 4 different dimensions (channels) into jpg image files containing one single stack as an optional channel.

module _random_forest.py_:

Random Forest classification model to find those jpg images that contain at least one stem cell colony. 

For testing there are unstacked jpg files in folder _train/jpg_. The module _random_forest.py_ will run a training on these jpg files and eventually makes predictions on _test/jpg_.


## Data

Initial data format: tif files deriving from confocal microscopes

## To do:

- [ ] Making one main.py file that drives the whole process.

_"In programming, a “driver file” contains the “driver function,” commonly known as main. This function serves as the entry point for your program. In Python, a typical driver file might be named main.py, and within it, you'll define a main() function. The concept of driver files and functions is critical for structuring your code, especially in larger applications. Here, we will discuss the rules and exceptions specific to driver files and functions in Python."_

- [ ] Adding more supervised classification models

