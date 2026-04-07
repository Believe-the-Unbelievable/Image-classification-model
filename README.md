# Image-classification-model
Used Pre-trained model - EfficientNetB0

This model will predict whether the image given is 'Cat or Dog'
Which is trained on 17k+ dataset of cats and dogs
The Dataset splitted into Train, Validation and Test sets using a script(train-70%, validation-15%, test-15%)

Dataset(24k+ images total) - 'https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset?select=PetImages' 

#Step 1:
install the requirement (pip install -r requirement.txt)

#Step 2:
Download the dataset and the dataset using 'data_split.py'

#Step 3:
Train the model using 'train.py'

#Step 4:
Then run the 'evaluate.py' for finding the accuracy on Test set

#Final Step:
To predict : run 'python predict.py --image <img path>'    
