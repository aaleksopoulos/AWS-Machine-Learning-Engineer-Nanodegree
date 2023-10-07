**NOTE:** This file is a template that you can use to create the README for your project. The **TODO** comments below will highlight the information you should be sure to include.

# Your Project Title Here

**TODO:** Write a short introduction to your project.

## Project Set Up and Installation
For this project, a SageMaker notebook was used, using ```ml.t3.medium``` as instance type, and a jupyter notebook with ```conda_amazonei_pytorch_latest_p37``` kernel. The isntance type seems to be more than efficient for the work that we will perform in the notebook [basic data manipulation], since all the training will be performed in estimators. Also, that specific instance type is eligible for free tier, reducing the cost.

## Dataset

### Overview
The dataset that will be used in this project, the [Amazon Bin Image Dataset](https://registry.opendata.aws/amazon-bin-imagery/) is provided by Amazon. It contains over 500,000 images and metadata from bins of a pod in an operating Amazon Fulfillment Center. The bin images in this dataset are captured as robot units carry pods as part of normal Amazon Fulfillment Center operations, and is under Creative Commons Attribution-NonCommercial-ShareAlike 3.0 United States License. Ιmages are located in the bin-images directory, and metadata for each image is located in the metadata directory. Images and their associated metadata share simple numerical unique identifiers. An example image and it's corresponding metadata follows  
*Example Image*  
![Example image](images/523.jpg)  
*Metadata*  
![Metadata](images/metadata.png)  

### Access
In order to download the dataset, the instructor's provided the file ```file_list.json```, with which a subset of them is downloaded, and organized in subfolders, with the name of each subfolder corresponding to the number of items that exist in each bin. Then, in order to create training, testing and validation sets to proper function the ML pipeline, the dataset was splitted and 60% of them consisted the training set, 20% test and 20% validation, which were uploaded to S3, using a boto3 client.

## Model Training
For this project I chose the ResNet50 model, which is a popular convolutional neural network (CNN) that is 50 layers deep, and has performed well in computer vision tasks. That model is pretrained, and a training part was added to it, in order to be used for the given case. The extra part consists of a fully connected neural network with the following configuration

```
    model.fc = nn.Sequential(
        nn.Linear(num_features, 254),
        nn.ReLU(),
        nn.Linear(254, 128),
        nn.ReLU(),
        nn.Linear(128, num_out_classes))
```

## Machine Learning Pipeline
The first step is to download the dataset, split it in Training-Testing-Validation sets and upload them to S3.  
Next, we performed hyperparameter optimization, in order to define the best possible combination of some choses hyperparamters. We chose the following to optimize
1. learning rate: Determins the step size of each iteration. If the value is large, it might miss the minimun of the loss function, if it is small will need higher number of iterations for the algorithm to converge
2. batch size: the number of samples that will be used in the network. Typically, neural networks train faster with mini-batches. The advantage of using smaller batch is that it requirres fewer resources, but it may miss the minimum of the loss function.
3. epochs: number of complete iterations of the neural network. If the number is low it may not be sufficient to find the minimum, if it is high it might lead to overfit [the model to memorize the dataset and cannot generalize]
After we found the best combination of the hyperparameters, we trained our model using them, deployed it and was ready to be used for inferencing.  
Finally, we used multiinstance training, which uses more instances to train the model, thus it trains faster but it costs more.

## Standout Suggestions
1. HyperParameter Optimization
2. Model Deployment [had some issues with the inference, any help is welcomed!! Regarding that error, it had occured to me and in the 3rd project, although for a reason I did not understand it resolved by itself. I had issued a question in the Udacity platform and still got no answers]
3. DEbugger and profiler report
4. Multi-instance training
