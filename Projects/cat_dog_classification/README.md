# References:

- https://data-flair.training/blogs/cats-dogs-classification-deep-learning-project-beginners/


# Steps to run this project:

1) Run the "cat_dog_classification_training.py" script for training the CNN keras model on cat dog classification dataset provided in the project.

NOTE: you can play with this script by changing the optimizer and adding/removing the layers of model or changing the epochs and batch size. You can also showcase the model performance using confusion matrix.

2) After completion of training, "model_catsVSdogs_10epoch.h5" this h5 model file is generated and this is then used in inference script that is "Inference.py" to showcase the results of the images in the "test" folder.

$ python Inference.py

Output:
 cat.4010.jpg ----->	Predicted:  dog 	Actual:  cat

 cat.4052.jpg ----->	Predicted:  cat 	Actual:  cat

 dog.4888.jpg ----->	Predicted:  cat 	Actual:  dog

 dog.4962.jpg ----->	Predicted:  dog 	Actual:  dog

 cat.4018.jpg ----->	Predicted:  cat 	Actual:  cat

 dog.4853.jpg ----->	Predicted:  dog 	Actual:  dog

 cat.4023.jpg ----->	Predicted:  cat 	Actual:  cat

 dog.4808.jpg ----->	Predicted:  cat 	Actual:  dog

3) Using Tkinter, app is developed just to get introduced with the Tkinter showcasing the inference rersults.

$ python Tkinter_inference_app.py
