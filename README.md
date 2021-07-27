# Custom-Build-OCR-Engine
Custom build OCR rengine to detect the text from the images.

The project consists of 2 python files.
1. data_genereation.py is for creating the data from the images for training and testing the data.
2. OCR_trainig_testing.py is for training and testing on the generated data using KNN algorithm and predict the text in new images.

The data folder consists of the images of alphabets and different fonts which are used in this project.

To run this project follow the below steps

Data generation file
1. Change the paths of the images that are used in the code and make you local path of the images saved.
2. run the data generator python file by any editor or teminal of your choice.
3. one image box will open with the image of text you are using to read inthe first steps of code and there will be coordinates drawn on particular characteer in that image, you need to type the character in key board like for example if you are reading an image consisting of alphabets A-z and the charqacter highlited is 'A' you need type the character A in your keyboard. make sure to type all the characters in the image and correctly. Once its done the image panel will automatically close and the data files will be genereated to the path you will mention in last lines of code.

OCR_Engine
1. Here you need to use the genereted data fioles to read and train to the KNN Algorithm and machine will read the data and gives the out of the detected text of the given image in a different panel.

Play around with different images. 

Please feel free to contact e through comments if you need futher assistance.
