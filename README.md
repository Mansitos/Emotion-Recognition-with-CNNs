# Emotion-Recognition-with-CNNs

This repository is about an university assignement from the Deep Learning Course (Università degli Studi di Udine).

Master’s Degree in Computer Science, University of Udine
Course of Deep Learning, Academic Year 2021/2022

Course held by: Prof. Giuseppe Serra, Dr. Beatrice Portelli,
Dr. Giovanni D’Agostino
Dipartimento di Scienze Matematiche, Informatiche e Fisiche

# Students: Andrea Mansi (137857) & Christian Cagnoni (137690)

-----------------------------------------------------------------
-----------------------------------------------------------------

```python
import os
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plot
from PIL import Image
import torchvision.transforms as T
import time
```

## 1. Data import and preprocessing

The input dataset is from FER2013 competition.

Dataset URL: https://www.kaggle.com/ashishpatel26/facial-expression-recognitionferchallenge


```python
# Data reading from csv file
# Insert your path here:
data_path = "C:/Users/Mansitos_Picci/Desktop/fer2013.csv"
dataframe = pd.read_csv(data_path)

# Printing dataset main infos
print("Shape:" + str(dataframe.shape))
print(dataframe.head(4))
```

    Shape:(35887, 3)
       emotion                                             pixels     Usage
    0        0  70 80 82 72 58 58 60 63 54 58 60 48 89 115 121...  Training
    1        0  151 150 147 155 148 133 111 140 170 174 182 15...  Training
    2        2  231 212 156 164 174 138 161 173 182 200 106 38...  Training
    3        4  24 32 36 30 32 23 19 20 30 41 21 22 32 34 21 1...  Training
    

Counting and plotting all the emotions. Dataset is not "balanced" some emotions are less present.


```python
emotions_labels_to_text = {0:'Anger >:(', 1:'Disgust :S', 2:'Fear :"(', 3:'Happiness :D', 4: 'Sadness :(', 5: 'Surprise :O', 6: 'Neutral :|'}

emotions_counts = dataframe['emotion'].value_counts(sort=False).reset_index()
emotions_counts.columns = ['emotion', 'number']
emotions_counts['emotion'] = emotions_counts['emotion'].map(emotions_labels_to_text)

plot.figure(figsize=(10, 4))
plot.bar(emotions_counts.emotion, emotions_counts.number)
plot.title('Emotions distribution')
plot.xlabel('\nEmotions', fontsize=12)
plot.ylabel('Number\n', fontsize=12)
plot.show()
```


    
![png]("images/output_5_0.png")
    


Processing the input data and preparing test and validation sets. Public test will be used, private test will be ignored.


```python
# Taking Training data (dataset is already subdivided into training and test (private and public))
data_train = dataframe[dataframe.Usage == "Training"]
data_test  = dataframe[dataframe.Usage == "PublicTest"]

# Processing the pixel column: converting the list (in which each pixel value is separated by a whitespace) into a list of ints
imgs_train = pd.DataFrame(data_train.pixels.str.split(" ").tolist())
imgs_test  = pd.DataFrame(data_test.pixels.str.split(" ").tolist())

# Float conversion
imgs_train = imgs_train.values.astype(float)
imgs_test = imgs_test.values.astype(float)

# reshaping to match model input
x_train = imgs_train.reshape(-1, 48 , 48 ,1)
x_test = imgs_test.reshape(-1, 48, 48, 1)

# taking label train column and converting into monodimensional array
y_train = data_train["emotion"].values
y_test = data_test["emotion"].values
```

Dataset augmentation phase. This phase enlarge the dataset by applying transformations to images. It can be skipped in order to use the raw input dataset. Enlarged dataset will require more time for training phase.


```python
# set to TRUE if you want the test dataset to be enlarged by applying transformations to images
enanche_dataset_flag = True
```


```python
if(enanche_dataset_flag):
    print("Executing dataset augmentation... phase: 1")
    img=[]

    def arrayToImage(array):
        new_im = Image.new("L", (48,48))

        x_offset = 0
        for im in array:
          new_im.paste(im, (x_offset,0))
          x_offset += im.size[0]

        img.append((new_im.rotate(-90)).transpose(Image.FLIP_LEFT_RIGHT))

    for i in range(len(x_train)):
        tmp_img=[]
        for j in range(len(x_train[i])):
            tmp_img.append(Image.fromarray(x_train[i][j]))
        arrayToImage(tmp_img)

    
    print("Number of starting images: " + str(len(img)))
    print("Showing a random starting image:")
    plot.imshow(img[0],cmap='gray')
else:
    print("Dataset augmentation skipped...")
```

    Executing dataset augmentation... phase: 1
    Number of starting images: 28709
    Showing a random starting image:
    


    
![png](images\output_10_1.png)
    



```python
if(enanche_dataset_flag):
    print("Executing dataset augmentation... phase: 2")
    torch.manual_seed(42)

    transformedImages=[]
    transformedLabels=[]

    perspective_transfromer = T.RandomPerspective(distortion_scale=0.5,p=0.5)
    rotater = T.RandomRotation(degrees=(-90, 90))
    
    for i in range(len(img)):
        transformedImages.append(perspective_transfromer(img[i]))
        transformedLabels.append(y_train[i])
    for i in range(len(img)):
        transformedImages.append(rotater(img[i]))
        transformedLabels.append(y_train[i])
```

    Executing dataset augmentation... phase: 2
    


```python
if(enanche_dataset_flag):
    x_train_ransformed=[]
    print("Executing dataset augmentation... phase: 3")

    for i in transformedImages:
        x_train_ransformed.append((np.asanyarray(i)).reshape(48,48,1))

    x_train = np.concatenate((x_train,np.array(x_train_ransformed)))
    y_train = np.concatenate((y_train,np.array(transformedLabels)))
    
    print("Number of final images: " + str(len(x_train)))
    print("Showing a random transformed image:")
    plot.imshow(transformedImages[-1],cmap='gray')
```

    Executing dataset augmentation... phase: 3
    Number of final images: 86127
    Showing a random transformed image:
    


    
![png](images\output_12_1.png)
    


Normalizing the emotions img. count. Dataset is not balanced, some emotions have a lower abs. frequency compared to others (disgust is the extreme example of this).

This phase is optionally executed. It takes the abs. frequency of the less frequent emotion and randomly discards imgs. from other classes in order to have the same amount of input data for each emotion. 

Note that this can improve accuracy because input data will be more balanced but at the same time some information is lost (so accuracy can decrease).


```python
minAbsFreq=min(data_train["emotion"].value_counts())
originalShape=imgs_train.shape

number_of_applied_transformations = 3
originalShapeAugmented=(originalShape[0]*number_of_applied_transformations,originalShape[1])
lowerBound=minAbsFreq*number_of_applied_transformations

x_train_norm=x_train.reshape(originalShapeAugmented)

tmpSer=[]
merge=""
old_percentage = 0

for i in range(len(x_train)):
    new_percentage = int((i/len(x_train_norm))*100)
    if(old_percentage < new_percentage):
        print("Status:" + str(new_percentage) + "%")
        old_percentage = new_percentage
    for j in range(len(x_train[i])):
        merge=merge+" "+str(x_train_norm[i][j])
    tmpSer.append(merge)
    merge=""
print("Status: 100%")
ser=pd.Series(tmpSer)

d={'emotion':y_train,'pixels':ser}
augmentedDF=pd.DataFrame(data=d)
```

    Status:1%
    Status:2%
    Status:3%
    Status:4%
    Status:5%
    Status:6%
    Status:7%
    Status:8%
    Status:9%
    Status:10%
    Status:11%
    Status:12%
    Status:13%
    Status:14%
    Status:15%
    Status:16%
    Status:17%
    Status:18%
    Status:19%
    Status:20%
    Status:21%
    Status:22%
    Status:23%
    Status:24%
    Status:25%
    Status:26%
    Status:27%
    Status:28%
    Status:29%
    Status:30%
    Status:31%
    Status:32%
    Status:33%
    Status:34%
    Status:35%
    Status:36%
    Status:37%
    Status:38%
    Status:39%
    Status:40%
    Status:41%
    Status:42%
    Status:43%
    Status:44%
    Status:45%
    Status:46%
    Status:47%
    Status:48%
    Status:49%
    Status:50%
    Status:51%
    Status:52%
    Status:53%
    Status:54%
    Status:55%
    Status:56%
    Status:57%
    Status:58%
    Status:59%
    Status:60%
    Status:61%
    Status:62%
    Status:63%
    Status:64%
    Status:65%
    Status:66%
    Status:67%
    Status:68%
    Status:69%
    Status:70%
    Status:71%
    Status:72%
    Status:73%
    Status:74%
    Status:75%
    Status:76%
    Status:77%
    Status:78%
    Status:79%
    Status:80%
    Status:81%
    Status:82%
    Status:83%
    Status:84%
    Status:85%
    Status:86%
    Status:87%
    Status:88%
    Status:89%
    Status:90%
    Status:91%
    Status:92%
    Status:93%
    Status:94%
    Status:95%
    Status:96%
    Status:97%
    Status:98%
    Status:99%
    Status: 100%
    


```python

```

Creating model input: data is converted into tensors and then loaded into 2 dataloaders: train and test dataloaders.


```python
tensor_ids_train = torch.Tensor(x_train)
labels_train = torch.LongTensor(y_train)
train_dataset = TensorDataset(tensor_ids_train, labels_train)

tensor_ids_test = torch.Tensor(x_test)
labels_test = torch.LongTensor(y_test)
test_dataset = TensorDataset(tensor_ids_test, labels_test)
```


```python
batch_size = 128

train_dataloader = DataLoader(train_dataset,
                              sampler = RandomSampler(train_dataset),
                              batch_size = batch_size)
test_dataloader = DataLoader(test_dataset,
                              sampler = RandomSampler(test_dataset),
                              batch_size = batch_size)
```

Defining useful functions for tensor printing


```python
def showImage(image,size,index,description):
    img = image.reshape(size,size)
    plot.axis("off")
    #print(description)
    plot.figure(index)
    plot.imshow(img, cmap ='gray')

# This function can be used to see a tensor as an image
def showTensorAsImage(tensor,size,index,description):
      tensor_copy = tensor.to(torch.device("cpu"))
      tensor_copy = tensor_copy.detach()
      showImage(np.array(tensor_copy[0][0]),size,index,description)
     
```

Checking machine devices. Enable CUDA computation if available.


```python
print("Is GPU available?")
if(torch.cuda.is_available()):
    print("Yes :)")
else:
    print("No :(")
print("")
print("Is cudnn backend enabled?")
print(torch.backends.cudnn.enabled)
print("")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training will run on device: " + str(device))
if(str(device)=="cuda:0"):print( torch.cuda.get_device_name(0))
```

    Is GPU available?
    Yes :)
    
    Is cudnn backend enabled?
    True
    
    Training will run on device: cuda:0
    NVIDIA GeForce RTX 3070 Ti
    

## 2. Defining the Model

The following model is the last iteration/version. Previous versions are not included but informations are included in the PDF report.


```python
class Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(1,80,kernel_size=5,padding=2) # First convolutional layer
      self.conv1_bn = nn.BatchNorm2d(80)
      self.pool = nn.MaxPool2d(kernel_size=2,stride=2) # Pooling layer
      self.conv2 = nn.Conv2d(80,140,kernel_size=5,padding=2) # Second convolutional layer
      self.conv2_bn = nn.BatchNorm2d(140)
      self.conv3 = nn.Conv2d(140,200,kernel_size=3,padding=1) # Third convolutional layer
      self.conv3_bn = nn.BatchNorm2d(200)
      self.conv4 = nn.Conv2d(200,250,kernel_size=3,padding=1) # Fourth convolutional layer
      self.conv4_bn = nn.BatchNorm2d(250)
      self.fc1 = nn.Linear(250*3*3,160) # First linear layer
      self.fc1_bn = nn.BatchNorm1d(160)
      self.fc2 = nn.Linear(160,70) # Second linear layer
      self.fc2_bn = nn.BatchNorm1d(70)
      self.fc3 = nn.Linear(70,20) # Third linear layer
      self.fc3_bn = nn.BatchNorm1d(20)
      self.relu = nn.ReLU()
      self.softmax = nn.Softmax(dim=1)
      self.drop = nn.Dropout(p=0.25)

    # Applying step-by-step the image classification architecture to the input
    def forward(self, x):     
      x = self.conv1(x) 
      x = self.conv1_bn(x)
      x = self.relu(x)
      x = self.pool(x)
      x = self.drop(x)
        
      if(net.training == False):
        showTensorAsImage(x,24,1,description="Random tensor from 1st conv. layer")
        
      x = self.conv2(x)  
      x = self.conv2_bn(x)
      x = self.relu(x)
      x = self.pool(x)
      x = self.drop(x)
    
      if(net.training == False):
        showTensorAsImage(x,12,2,description="Random tensor from 2nd conv. layer")
    
      x = self.conv3(x)
      x = self.conv3_bn(x)
      x = self.relu(x)
      x = self.pool(x)
      x = self.drop(x)
        
      if(net.training == False):
        showTensorAsImage(x,6,3,description="Random tensor from 3rd conv. layer")
        
      x = self.conv4(x)
      x = self.conv4_bn(x)
      x = self.relu(x)
      x = self.pool(x)
      x = self.drop(x)
    
      x = x.reshape(x.size(0),250*3*3) # Modifying the shape of x to make the tensor fit in the first linear
      x = self.fc1(x)
      x = self.fc1_bn(x)
      x = self.relu(x) # Applying the activation function
      x = self.drop(x)
    
      x = self.fc2(x)
      x = self.fc2_bn(x)
      x = self.relu(x) # Applying the activation function
      x = self.drop(x)
        
      x = self.fc3(x)
      x = self.fc3_bn(x)
      x = self.softmax(x)
      return x

net = Net()
print(net)
```

    Net(
      (conv1): Conv2d(1, 80, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (conv1_bn): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv2): Conv2d(80, 140, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (conv2_bn): BatchNorm2d(140, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(140, 200, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv3_bn): BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv4): Conv2d(200, 250, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv4_bn): BatchNorm2d(250, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc1): Linear(in_features=2250, out_features=160, bias=True)
      (fc1_bn): BatchNorm1d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc2): Linear(in_features=160, out_features=70, bias=True)
      (fc2_bn): BatchNorm1d(70, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc3): Linear(in_features=70, out_features=20, bias=True)
      (fc3_bn): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (softmax): Softmax(dim=1)
      (drop): Dropout(p=0.25, inplace=False)
    )
    

#### Train function definition


```python
# Moving model to device (to cuda if available)
net = net.to(device)
net.train()

criterion = nn.CrossEntropyLoss() # Defining the criterion

#optimizer = optim.SGD(net.parameters(),momentum=0.9,lr=0.001) # Defining the optimizer
optimizer = optim.Adam(net.parameters(),lr=0.003)

start_time = time.time()

for epoch in range(350): #Looping over the dataset three times

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        
        inputs, labels = data # The input data is a list [inputs, labels]

        inputs = inputs.permute(0, 3, 1, 2) # permuting the input to match the order used by pytorch
        
        # Moving to device (to cuda if available)
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad() # Setting the parameter gradients to zero

        outputs = net(inputs) # Forward pass
        loss = criterion(outputs,labels) # Applying the criterion
        loss.backward() # Backward pass
        optimizer.step() # Optimization step

        running_loss += loss.item() # Updating the running loss
        if i % len(train_dataloader) == len(train_dataloader)-1:  # Printing the running loss
            print('[epoch: %d, mini-batch: %d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / len(train_dataloader)))
            running_loss = 0.0

print('Finished Training :)')
print("Training time: %s seconds" % (time.time() - start_time))
```

    [epoch: 1, mini-batch: 673] loss: 2.790
    [epoch: 2, mini-batch: 673] loss: 2.666
    [epoch: 3, mini-batch: 673] loss: 2.630
    [epoch: 4, mini-batch: 673] loss: 2.604
    [epoch: 5, mini-batch: 673] loss: 2.584
    [epoch: 6, mini-batch: 673] loss: 2.569
    [epoch: 7, mini-batch: 673] loss: 2.561
    [epoch: 8, mini-batch: 673] loss: 2.547
    [epoch: 9, mini-batch: 673] loss: 2.536
    [epoch: 10, mini-batch: 673] loss: 2.529
    [epoch: 11, mini-batch: 673] loss: 2.525
    [epoch: 12, mini-batch: 673] loss: 2.515
    [epoch: 13, mini-batch: 673] loss: 2.510
    [epoch: 14, mini-batch: 673] loss: 2.504
    [epoch: 15, mini-batch: 673] loss: 2.499
    [epoch: 16, mini-batch: 673] loss: 2.493
    [epoch: 17, mini-batch: 673] loss: 2.488
    [epoch: 18, mini-batch: 673] loss: 2.484
    [epoch: 19, mini-batch: 673] loss: 2.481
    [epoch: 20, mini-batch: 673] loss: 2.477
    [epoch: 21, mini-batch: 673] loss: 2.473
    [epoch: 22, mini-batch: 673] loss: 2.470
    [epoch: 23, mini-batch: 673] loss: 2.467
    [epoch: 24, mini-batch: 673] loss: 2.464
    [epoch: 25, mini-batch: 673] loss: 2.461
    [epoch: 26, mini-batch: 673] loss: 2.457
    [epoch: 27, mini-batch: 673] loss: 2.456
    [epoch: 28, mini-batch: 673] loss: 2.451
    [epoch: 29, mini-batch: 673] loss: 2.449
    [epoch: 30, mini-batch: 673] loss: 2.447
    [epoch: 31, mini-batch: 673] loss: 2.445
    [epoch: 32, mini-batch: 673] loss: 2.442
    [epoch: 33, mini-batch: 673] loss: 2.439
    [epoch: 34, mini-batch: 673] loss: 2.439
    [epoch: 35, mini-batch: 673] loss: 2.435
    [epoch: 36, mini-batch: 673] loss: 2.435
    [epoch: 37, mini-batch: 673] loss: 2.432
    [epoch: 38, mini-batch: 673] loss: 2.431
    [epoch: 39, mini-batch: 673] loss: 2.429
    [epoch: 40, mini-batch: 673] loss: 2.426
    [epoch: 41, mini-batch: 673] loss: 2.426
    [epoch: 42, mini-batch: 673] loss: 2.423
    [epoch: 43, mini-batch: 673] loss: 2.420
    [epoch: 44, mini-batch: 673] loss: 2.419
    [epoch: 45, mini-batch: 673] loss: 2.418
    [epoch: 46, mini-batch: 673] loss: 2.415
    [epoch: 47, mini-batch: 673] loss: 2.414
    [epoch: 48, mini-batch: 673] loss: 2.414
    [epoch: 49, mini-batch: 673] loss: 2.413
    [epoch: 50, mini-batch: 673] loss: 2.409
    [epoch: 51, mini-batch: 673] loss: 2.410
    [epoch: 52, mini-batch: 673] loss: 2.407
    [epoch: 53, mini-batch: 673] loss: 2.405
    [epoch: 54, mini-batch: 673] loss: 2.406
    [epoch: 55, mini-batch: 673] loss: 2.403
    [epoch: 56, mini-batch: 673] loss: 2.403
    [epoch: 57, mini-batch: 673] loss: 2.401
    [epoch: 58, mini-batch: 673] loss: 2.398
    [epoch: 59, mini-batch: 673] loss: 2.397
    [epoch: 60, mini-batch: 673] loss: 2.397
    [epoch: 61, mini-batch: 673] loss: 2.395
    [epoch: 62, mini-batch: 673] loss: 2.394
    [epoch: 63, mini-batch: 673] loss: 2.394
    [epoch: 64, mini-batch: 673] loss: 2.391
    [epoch: 65, mini-batch: 673] loss: 2.392
    [epoch: 66, mini-batch: 673] loss: 2.391
    [epoch: 67, mini-batch: 673] loss: 2.390
    [epoch: 68, mini-batch: 673] loss: 2.388
    [epoch: 69, mini-batch: 673] loss: 2.385
    [epoch: 70, mini-batch: 673] loss: 2.388
    [epoch: 71, mini-batch: 673] loss: 2.385
    [epoch: 72, mini-batch: 673] loss: 2.383
    [epoch: 73, mini-batch: 673] loss: 2.384
    [epoch: 74, mini-batch: 673] loss: 2.382
    [epoch: 75, mini-batch: 673] loss: 2.381
    [epoch: 76, mini-batch: 673] loss: 2.382
    [epoch: 77, mini-batch: 673] loss: 2.378
    [epoch: 78, mini-batch: 673] loss: 2.379
    [epoch: 79, mini-batch: 673] loss: 2.379
    [epoch: 80, mini-batch: 673] loss: 2.378
    [epoch: 81, mini-batch: 673] loss: 2.375
    [epoch: 82, mini-batch: 673] loss: 2.375
    [epoch: 83, mini-batch: 673] loss: 2.375
    [epoch: 84, mini-batch: 673] loss: 2.375
    [epoch: 85, mini-batch: 673] loss: 2.371
    [epoch: 86, mini-batch: 673] loss: 2.371
    [epoch: 87, mini-batch: 673] loss: 2.371
    [epoch: 88, mini-batch: 673] loss: 2.370
    [epoch: 89, mini-batch: 673] loss: 2.371
    [epoch: 90, mini-batch: 673] loss: 2.368
    [epoch: 91, mini-batch: 673] loss: 2.371
    [epoch: 92, mini-batch: 673] loss: 2.368
    [epoch: 93, mini-batch: 673] loss: 2.366
    [epoch: 94, mini-batch: 673] loss: 2.367
    [epoch: 95, mini-batch: 673] loss: 2.364
    [epoch: 96, mini-batch: 673] loss: 2.364
    [epoch: 97, mini-batch: 673] loss: 2.361
    [epoch: 98, mini-batch: 673] loss: 2.361
    [epoch: 99, mini-batch: 673] loss: 2.361
    [epoch: 100, mini-batch: 673] loss: 2.362
    [epoch: 101, mini-batch: 673] loss: 2.360
    [epoch: 102, mini-batch: 673] loss: 2.359
    [epoch: 103, mini-batch: 673] loss: 2.359
    [epoch: 104, mini-batch: 673] loss: 2.358
    [epoch: 105, mini-batch: 673] loss: 2.358
    [epoch: 106, mini-batch: 673] loss: 2.356
    [epoch: 107, mini-batch: 673] loss: 2.357
    [epoch: 108, mini-batch: 673] loss: 2.357
    [epoch: 109, mini-batch: 673] loss: 2.354
    [epoch: 110, mini-batch: 673] loss: 2.356
    [epoch: 111, mini-batch: 673] loss: 2.352
    [epoch: 112, mini-batch: 673] loss: 2.354
    [epoch: 113, mini-batch: 673] loss: 2.351
    [epoch: 114, mini-batch: 673] loss: 2.352
    [epoch: 115, mini-batch: 673] loss: 2.351
    [epoch: 116, mini-batch: 673] loss: 2.350
    [epoch: 117, mini-batch: 673] loss: 2.349
    [epoch: 118, mini-batch: 673] loss: 2.350
    [epoch: 119, mini-batch: 673] loss: 2.349
    [epoch: 120, mini-batch: 673] loss: 2.348
    [epoch: 121, mini-batch: 673] loss: 2.346
    [epoch: 122, mini-batch: 673] loss: 2.348
    [epoch: 123, mini-batch: 673] loss: 2.345
    [epoch: 124, mini-batch: 673] loss: 2.347
    [epoch: 125, mini-batch: 673] loss: 2.345
    [epoch: 126, mini-batch: 673] loss: 2.345
    [epoch: 127, mini-batch: 673] loss: 2.345
    [epoch: 128, mini-batch: 673] loss: 2.342
    [epoch: 129, mini-batch: 673] loss: 2.342
    [epoch: 130, mini-batch: 673] loss: 2.343
    [epoch: 131, mini-batch: 673] loss: 2.340
    [epoch: 132, mini-batch: 673] loss: 2.343
    [epoch: 133, mini-batch: 673] loss: 2.341
    [epoch: 134, mini-batch: 673] loss: 2.340
    [epoch: 135, mini-batch: 673] loss: 2.339
    [epoch: 136, mini-batch: 673] loss: 2.339
    [epoch: 137, mini-batch: 673] loss: 2.339
    [epoch: 138, mini-batch: 673] loss: 2.338
    [epoch: 139, mini-batch: 673] loss: 2.339
    [epoch: 140, mini-batch: 673] loss: 2.337
    [epoch: 141, mini-batch: 673] loss: 2.337
    [epoch: 142, mini-batch: 673] loss: 2.336
    [epoch: 143, mini-batch: 673] loss: 2.335
    [epoch: 144, mini-batch: 673] loss: 2.337
    [epoch: 145, mini-batch: 673] loss: 2.334
    [epoch: 146, mini-batch: 673] loss: 2.337
    [epoch: 147, mini-batch: 673] loss: 2.333
    [epoch: 148, mini-batch: 673] loss: 2.336
    [epoch: 149, mini-batch: 673] loss: 2.334
    [epoch: 150, mini-batch: 673] loss: 2.332
    [epoch: 151, mini-batch: 673] loss: 2.332
    [epoch: 152, mini-batch: 673] loss: 2.332
    [epoch: 153, mini-batch: 673] loss: 2.330
    [epoch: 154, mini-batch: 673] loss: 2.331
    [epoch: 155, mini-batch: 673] loss: 2.332
    [epoch: 156, mini-batch: 673] loss: 2.330
    [epoch: 157, mini-batch: 673] loss: 2.331
    [epoch: 158, mini-batch: 673] loss: 2.327
    [epoch: 159, mini-batch: 673] loss: 2.327
    [epoch: 160, mini-batch: 673] loss: 2.327
    [epoch: 161, mini-batch: 673] loss: 2.330
    [epoch: 162, mini-batch: 673] loss: 2.326
    [epoch: 163, mini-batch: 673] loss: 2.326
    [epoch: 164, mini-batch: 673] loss: 2.327
    [epoch: 165, mini-batch: 673] loss: 2.326
    [epoch: 166, mini-batch: 673] loss: 2.327
    [epoch: 167, mini-batch: 673] loss: 2.326
    [epoch: 168, mini-batch: 673] loss: 2.325
    [epoch: 169, mini-batch: 673] loss: 2.324
    [epoch: 170, mini-batch: 673] loss: 2.323
    [epoch: 171, mini-batch: 673] loss: 2.321
    [epoch: 172, mini-batch: 673] loss: 2.322
    [epoch: 173, mini-batch: 673] loss: 2.323
    [epoch: 174, mini-batch: 673] loss: 2.320
    [epoch: 175, mini-batch: 673] loss: 2.320
    [epoch: 176, mini-batch: 673] loss: 2.322
    [epoch: 177, mini-batch: 673] loss: 2.321
    [epoch: 178, mini-batch: 673] loss: 2.320
    [epoch: 179, mini-batch: 673] loss: 2.321
    [epoch: 180, mini-batch: 673] loss: 2.319
    [epoch: 181, mini-batch: 673] loss: 2.320
    [epoch: 182, mini-batch: 673] loss: 2.318
    [epoch: 183, mini-batch: 673] loss: 2.319
    [epoch: 184, mini-batch: 673] loss: 2.319
    [epoch: 185, mini-batch: 673] loss: 2.318
    [epoch: 186, mini-batch: 673] loss: 2.316
    [epoch: 187, mini-batch: 673] loss: 2.318
    [epoch: 188, mini-batch: 673] loss: 2.313
    [epoch: 189, mini-batch: 673] loss: 2.315
    [epoch: 190, mini-batch: 673] loss: 2.317
    [epoch: 191, mini-batch: 673] loss: 2.317
    [epoch: 192, mini-batch: 673] loss: 2.315
    [epoch: 193, mini-batch: 673] loss: 2.314
    [epoch: 194, mini-batch: 673] loss: 2.315
    [epoch: 195, mini-batch: 673] loss: 2.316
    [epoch: 196, mini-batch: 673] loss: 2.313
    [epoch: 197, mini-batch: 673] loss: 2.314
    [epoch: 198, mini-batch: 673] loss: 2.315
    [epoch: 199, mini-batch: 673] loss: 2.313
    [epoch: 200, mini-batch: 673] loss: 2.314
    [epoch: 201, mini-batch: 673] loss: 2.312
    [epoch: 202, mini-batch: 673] loss: 2.313
    [epoch: 203, mini-batch: 673] loss: 2.311
    [epoch: 204, mini-batch: 673] loss: 2.312
    [epoch: 205, mini-batch: 673] loss: 2.308
    [epoch: 206, mini-batch: 673] loss: 2.310
    [epoch: 207, mini-batch: 673] loss: 2.311
    [epoch: 208, mini-batch: 673] loss: 2.311
    [epoch: 209, mini-batch: 673] loss: 2.310
    [epoch: 210, mini-batch: 673] loss: 2.307
    [epoch: 211, mini-batch: 673] loss: 2.309
    [epoch: 212, mini-batch: 673] loss: 2.310
    [epoch: 213, mini-batch: 673] loss: 2.309
    [epoch: 214, mini-batch: 673] loss: 2.308
    [epoch: 215, mini-batch: 673] loss: 2.308
    [epoch: 216, mini-batch: 673] loss: 2.308
    [epoch: 217, mini-batch: 673] loss: 2.307
    [epoch: 218, mini-batch: 673] loss: 2.309
    [epoch: 219, mini-batch: 673] loss: 2.308
    [epoch: 220, mini-batch: 673] loss: 2.306
    [epoch: 221, mini-batch: 673] loss: 2.306
    [epoch: 222, mini-batch: 673] loss: 2.306
    [epoch: 223, mini-batch: 673] loss: 2.305
    [epoch: 224, mini-batch: 673] loss: 2.305
    [epoch: 225, mini-batch: 673] loss: 2.304
    [epoch: 226, mini-batch: 673] loss: 2.305
    [epoch: 227, mini-batch: 673] loss: 2.305
    [epoch: 228, mini-batch: 673] loss: 2.304
    [epoch: 229, mini-batch: 673] loss: 2.303
    [epoch: 230, mini-batch: 673] loss: 2.304
    [epoch: 231, mini-batch: 673] loss: 2.303
    [epoch: 232, mini-batch: 673] loss: 2.302
    [epoch: 233, mini-batch: 673] loss: 2.301
    [epoch: 234, mini-batch: 673] loss: 2.303
    [epoch: 235, mini-batch: 673] loss: 2.304
    [epoch: 236, mini-batch: 673] loss: 2.302
    [epoch: 237, mini-batch: 673] loss: 2.302
    [epoch: 238, mini-batch: 673] loss: 2.301
    [epoch: 239, mini-batch: 673] loss: 2.301
    [epoch: 240, mini-batch: 673] loss: 2.300
    [epoch: 241, mini-batch: 673] loss: 2.301
    [epoch: 242, mini-batch: 673] loss: 2.301
    [epoch: 243, mini-batch: 673] loss: 2.297
    [epoch: 244, mini-batch: 673] loss: 2.299
    [epoch: 245, mini-batch: 673] loss: 2.299
    [epoch: 246, mini-batch: 673] loss: 2.299
    [epoch: 247, mini-batch: 673] loss: 2.298
    [epoch: 248, mini-batch: 673] loss: 2.298
    [epoch: 249, mini-batch: 673] loss: 2.298
    [epoch: 250, mini-batch: 673] loss: 2.297
    [epoch: 251, mini-batch: 673] loss: 2.298
    [epoch: 252, mini-batch: 673] loss: 2.296
    [epoch: 253, mini-batch: 673] loss: 2.297
    [epoch: 254, mini-batch: 673] loss: 2.297
    [epoch: 255, mini-batch: 673] loss: 2.296
    [epoch: 256, mini-batch: 673] loss: 2.295
    [epoch: 257, mini-batch: 673] loss: 2.297
    [epoch: 258, mini-batch: 673] loss: 2.296
    [epoch: 259, mini-batch: 673] loss: 2.297
    [epoch: 260, mini-batch: 673] loss: 2.292
    [epoch: 261, mini-batch: 673] loss: 2.295
    [epoch: 262, mini-batch: 673] loss: 2.294
    [epoch: 263, mini-batch: 673] loss: 2.294
    [epoch: 264, mini-batch: 673] loss: 2.295
    [epoch: 265, mini-batch: 673] loss: 2.294
    [epoch: 266, mini-batch: 673] loss: 2.292
    [epoch: 267, mini-batch: 673] loss: 2.292
    [epoch: 268, mini-batch: 673] loss: 2.294
    [epoch: 269, mini-batch: 673] loss: 2.293
    [epoch: 270, mini-batch: 673] loss: 2.293
    [epoch: 271, mini-batch: 673] loss: 2.293
    [epoch: 272, mini-batch: 673] loss: 2.293
    [epoch: 273, mini-batch: 673] loss: 2.293
    [epoch: 274, mini-batch: 673] loss: 2.292
    [epoch: 275, mini-batch: 673] loss: 2.290
    [epoch: 276, mini-batch: 673] loss: 2.292
    [epoch: 277, mini-batch: 673] loss: 2.291
    [epoch: 278, mini-batch: 673] loss: 2.289
    [epoch: 279, mini-batch: 673] loss: 2.290
    [epoch: 280, mini-batch: 673] loss: 2.290
    [epoch: 281, mini-batch: 673] loss: 2.291
    [epoch: 282, mini-batch: 673] loss: 2.289
    [epoch: 283, mini-batch: 673] loss: 2.291
    [epoch: 284, mini-batch: 673] loss: 2.290
    [epoch: 285, mini-batch: 673] loss: 2.289
    [epoch: 286, mini-batch: 673] loss: 2.288
    [epoch: 287, mini-batch: 673] loss: 2.288
    [epoch: 288, mini-batch: 673] loss: 2.288
    [epoch: 289, mini-batch: 673] loss: 2.288
    [epoch: 290, mini-batch: 673] loss: 2.290
    [epoch: 291, mini-batch: 673] loss: 2.288
    [epoch: 292, mini-batch: 673] loss: 2.289
    [epoch: 293, mini-batch: 673] loss: 2.287
    [epoch: 294, mini-batch: 673] loss: 2.287
    [epoch: 295, mini-batch: 673] loss: 2.287
    [epoch: 296, mini-batch: 673] loss: 2.288
    [epoch: 297, mini-batch: 673] loss: 2.286
    [epoch: 298, mini-batch: 673] loss: 2.286
    [epoch: 299, mini-batch: 673] loss: 2.286
    [epoch: 300, mini-batch: 673] loss: 2.285
    [epoch: 301, mini-batch: 673] loss: 2.285
    [epoch: 302, mini-batch: 673] loss: 2.287
    [epoch: 303, mini-batch: 673] loss: 2.284
    [epoch: 304, mini-batch: 673] loss: 2.283
    [epoch: 305, mini-batch: 673] loss: 2.286
    [epoch: 306, mini-batch: 673] loss: 2.285
    [epoch: 307, mini-batch: 673] loss: 2.283
    [epoch: 308, mini-batch: 673] loss: 2.282
    [epoch: 309, mini-batch: 673] loss: 2.285
    [epoch: 310, mini-batch: 673] loss: 2.285
    [epoch: 311, mini-batch: 673] loss: 2.287
    [epoch: 312, mini-batch: 673] loss: 2.284
    [epoch: 313, mini-batch: 673] loss: 2.284
    [epoch: 314, mini-batch: 673] loss: 2.282
    [epoch: 315, mini-batch: 673] loss: 2.285
    [epoch: 316, mini-batch: 673] loss: 2.283
    [epoch: 317, mini-batch: 673] loss: 2.281
    [epoch: 318, mini-batch: 673] loss: 2.281
    [epoch: 319, mini-batch: 673] loss: 2.282
    [epoch: 320, mini-batch: 673] loss: 2.282
    [epoch: 321, mini-batch: 673] loss: 2.282
    [epoch: 322, mini-batch: 673] loss: 2.281
    [epoch: 323, mini-batch: 673] loss: 2.280
    [epoch: 324, mini-batch: 673] loss: 2.282
    [epoch: 325, mini-batch: 673] loss: 2.282
    [epoch: 326, mini-batch: 673] loss: 2.282
    [epoch: 327, mini-batch: 673] loss: 2.280
    [epoch: 328, mini-batch: 673] loss: 2.279
    [epoch: 329, mini-batch: 673] loss: 2.279
    [epoch: 330, mini-batch: 673] loss: 2.280
    [epoch: 331, mini-batch: 673] loss: 2.284
    [epoch: 332, mini-batch: 673] loss: 2.280
    [epoch: 333, mini-batch: 673] loss: 2.279
    [epoch: 334, mini-batch: 673] loss: 2.280
    [epoch: 335, mini-batch: 673] loss: 2.278
    [epoch: 336, mini-batch: 673] loss: 2.279
    [epoch: 337, mini-batch: 673] loss: 2.277
    [epoch: 338, mini-batch: 673] loss: 2.276
    [epoch: 339, mini-batch: 673] loss: 2.278
    [epoch: 340, mini-batch: 673] loss: 2.278
    [epoch: 341, mini-batch: 673] loss: 2.278
    [epoch: 342, mini-batch: 673] loss: 2.279
    [epoch: 343, mini-batch: 673] loss: 2.278
    [epoch: 344, mini-batch: 673] loss: 2.276
    [epoch: 345, mini-batch: 673] loss: 2.278
    [epoch: 346, mini-batch: 673] loss: 2.277
    [epoch: 347, mini-batch: 673] loss: 2.277
    [epoch: 348, mini-batch: 673] loss: 2.277
    [epoch: 349, mini-batch: 673] loss: 2.279
    [epoch: 350, mini-batch: 673] loss: 2.275
    Finished Training :)
    Training time: 19576.756263017654 seconds
    

#### Evaluating the model
Public test set is used in order to test che model accuracy and other metrics


```python
# Calculating the accuracy of the network on the whole dataset
correct = 0
total = 0
i = 0

# Setting network in evaluation/testing mode (for disabling dropout layers)
net.eval()

with torch.no_grad():
    for data in test_dataloader:
        
        images, labels = data # Getting the test data

        images = images.permute(0, 3, 1, 2)
        # Moving to device (cuda if available)
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images) # Getting the network output
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
accurcyMean=100 * correct / total
              
print('Accuracy of the network on the %d test images (Public test-set): %d %%' % (len(imgs_test), accurcyMean))

# showing accumulated (convs) plots
plot.show()
```

    Accuracy of the network on the 3589 test images (Public test-set): 66 %
    


    
![png](images\output_28_1.png)
    



    
![png](images\output_28_2.png)
    



    
![png](images\output_28_3.png)
    



```python
classes=len(np.unique(y_test))

#Calculating the accuracy of the network on each class of images
class_correct = list(0. for i in range(classes))
class_total = list(0. for i in range(classes))

net.eval()

with torch.no_grad():
    for data in test_dataloader:
        images, labels = data # Getting the test data

        images=images.permute(0,3,1,2)

        images=images.to(device)
        labels=labels.to(device)
        
        outputs = net(images) # Getting the network output
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(5):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
            if(class_total[label] == 0): print(":C")

accuracy=list(0. for i in range(classes))

for i in range(classes):
    accuracy[i] = 100 * (class_correct[i] / (class_total[i]+1))

for i in range(len(np.unique(y_test))):
    print('Accuracy of %s (Class: %s): \t %2d %%' % (emotions_labels_to_text[i],np.unique(y_test)[i], 100 * class_correct[i] / class_total[i]))
```

    Accuracy of Anger >:( (Class: 0): 	 76 %
    Accuracy of Disgust :S (Class: 1): 	  0 %
    Accuracy of Fear :"( (Class: 2): 	 45 %
    Accuracy of Happiness :D (Class: 3): 	 97 %
    Accuracy of Sadness :( (Class: 4): 	 52 %
    Accuracy of Surprise :O (Class: 5): 	 94 %
    Accuracy of Neutral :| (Class: 6): 	 75 %
    


    
![png](images\output_29_1.png)
    



    
![png](images\output_29_2.png)
    



    
![png](images\output_29_3.png)
    



```python
#Calculating the accuracy of the network on each class of images
class_true_positive = list(0. for i in range(classes))
class_false_positive = list(0. for i in range(classes))

class_true_negative = list(0. for i in range(classes))
class_false_negative = list(0. for i in range(classes))

net.eval()

for i in range(classes):
    for data in test_dataloader:
        images, labels = data # Getting the test data

        images=images.permute(0,3,1,2)

        images=images.to(device)
        labels=labels.to(device)
        
        outputs = net(images) # Getting the network output
        _, predicted = torch.max(outputs, 1)
        for j in range(len(labels)):
            if labels[j]==i:
                if predicted[j]==i:
                    class_true_positive[i]+=1
                else:
                    class_false_negative[i]+=1
            else:
                if predicted[j]==i:
                    class_false_positive[i]+=1
                    
precision = list(0. for i in range(classes))
recall = list(0. for i in range(classes))

for i in range(classes):
    precision[i]= 100 * class_true_positive[i] / (class_true_positive[i] + class_false_positive[i]+1)

for i in range(len(np.unique(y_test))):
    print('Precision of %5s : %2d %%' % (
        np.unique(y_test)[i], precision[i]))

for i in range(classes):
    recall[i]=100 * class_true_positive[i] / (class_true_positive[i] + class_false_negative[i]+1)

for i in range(len(np.unique(y_test))):
    print('Recall of %5s : %2d %%' % (
        np.unique(y_test)[i], recall[i]))

print("\n")

info={'recall':recall,'precision':precision,'accuracy':accuracy}
infoDF=pd.DataFrame(data=info)
print(infoDF)
print("Accuracy \t %d"%(accurcyMean))
```

    Precision of     0 : 55 %
    Precision of     1 :  0 %
    Precision of     2 : 59 %
    Precision of     3 : 83 %
    Precision of     4 : 55 %
    Precision of     5 : 81 %
    Precision of     6 : 58 %
    Recall of     0 : 61 %
    Recall of     1 :  0 %
    Recall of     2 : 45 %
    Recall of     3 : 85 %
    Recall of     4 : 60 %
    Recall of     5 : 79 %
    Recall of     6 : 63 %
    
    
          recall  precision   accuracy
    0  61.324786  55.728155  74.074074
    1   0.000000   0.000000   0.000000
    2  45.472837  59.788360  43.478261
    3  85.714286  83.027027  94.736842
    4  60.397554  55.633803  50.000000
    5  79.807692  81.173594  90.000000
    6  63.651316  58.814590  71.428571
    Accuracy 	 66
    


    
![png](images\output_30_1.png)
    



    
![png](images\output_30_2.png)
    



    
![png](images\output_30_3.png)
    

