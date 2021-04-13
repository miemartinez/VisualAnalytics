# Assignment 5 - CNNs on cultural image data
### Multi-class classification of impressionist painters
This repository contains all of the code and a subset of the data related to Assignment 5 for Visual Analytics.
The full data can be found on Kaggle:
https://www.kaggle.com/delayedkarma/impressionist-classifier-data

In the data folder there is a subset folder of the data for training and validating the model. The folders contain 10 individual folders for each artist with jpg files of their paintings.
The output of the python script is saved in the created out folder. This contains a txt file and a visualization saved as png of the model architecture. 
It also contains the development of loss and accuracy for the training and validation across epochs. Lastly, it contains the classification report that displays the model accuracy.

The script cnn_artists.py is in the src and it can be run without specifying any parameters. However, the user is able to define the filepaths to the training and validation data.
Furthermore, the user can define the number of epochs to train over and the batch size.
If nothing is chosen in the command line, defaults are set instead. <br>

__Parameters:__ <br>
```
    path2train: str <path-to-train-data>, default = "../data/subset/training"
    path2test: str <path-to-test-data>, default = "../data/subset/validation"
    n_epochs: int <number-of-epochs>, default = 20
    batch_size: int <batch-size>, default = 32
```
    
__Usage:__ <br>
```
    cnn-artists.py -t <path-to-train> -c <path-to-test-data> -n <number-of-epochs> -b <batch-size>
```
    
__Example:__ <br>
```
    $ python3 cnn-artists.py -t ../data/training/training -te ../data/validation/validation -n 30 -b 40
```

To ensure dependencies are in accordance with the ones used for the script, you can create the virtual environment "CNN_venv" by running the bash script create_cnn_venv.sh
```
    $ bash ./create_cnn_venv.sh
```
After creating the environment, you have to activate it. And then you can run the script with the dependencies:
```
    $ source CNN_venv/bin/activate
    $ cd src
    $ python3 cnn-artists.py
```
The outputs will appear in an out folder.


### Results:
For a view of the results when running the model on the whole data see the out folder.
The best result was obtain using a batch size of 32 and 20 epochs. Here, an accuracy score of 0.42 was observed when running it in the group. However, this didn't replicate when I ran it again on my own computer as can be seen in the out folder.
