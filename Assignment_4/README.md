# Assignment 4 - Classification benchmarks
This repository contains all of the code and data related to Assignment 4 for Visual Analytics.
The repository contains two scripts, i) logisitic regression classification and ii) neural network classification of the MNIST data.
Both scripts can be run without inputs (as they have defaults) or the user can specify these.
The output of the python script is also provided in the out folder in the data folder. This contains the classification metrics for both models saved as csv and a confusion matrix for the logistic regression model saved as png.

For use of the scripts see following: <br>

# lr-mnist.py <br>

__Parameters:__ <br>
```
    train_size: float <number-between-0-and-1>, default = 0.8
    classification_tolerance: float <number-between-0-and-1>, default = 0.1
    filename: str <choose-filename-for-csv>, default = "lr_classification_metrics.csv"
    path2image: str <path-to-unseen-image>
```
    
__Usage:__ <br>
```
    lr_mnist.py -t <train-size> -c <classification-tolerance> -f <chosen-filename> -p <path-to-image>
```
    
__Example:__ <br>
```
    $ python3 lr_mnist.py -t 0.7 -c 0.05 -p ../data/test.png
```


# nn-mnist.py <br>

__Parameters:__ <br>
```
    train_size: float <number-between-0-and-1>, default = 0.8
    filename: str <choose-filename-for-csv>, default = "nn_classification_metrics.csv"
```
    
__Usage:__ <br>
```
    nn_mnist.py -t <train-size> -f <chosen-filename>
```
    
__Example:__ <br>
```
    $ python3 nn_mnist.py -t 0.7 -f "nn_evaluation.csv"
```


# Dependencies
To ensure dependencies are in accordance with the ones used for the script, you can create the virtual environment "classifier_environment" by running the bash script create_classifier_venv.sh
```
    $ bash ./create_classifier_venv.sh
```
After creating the environment, you have to activate it. And then you can run the script with the dependencies:
```
    $ source classifier_environment/bin/activate
    $ cd src
    $ python3 lr_mnist.py -t 0.7 -c 0.05 -p ../data/test.png
```

The outputs will appear in the data/out folder.

OBS. both scripts run on the full MNIST data. This means that the nn-mnist.py takes a long time to run. To minimize running time you can substitute line 80-81 with:
```
    $ self.X = np.array(X[:1000])
    $ self.y = np.array(y[:1000])
```
This will result in a subset of only 1000 data points so it will run much faster (though the model might not be as accurate.
