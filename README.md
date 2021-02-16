# Image Captioning Project

### Adalberto Gonzalez

Computer Vision Nanodegree Program

# Overview
In this project, I've created a neural network architecture to automatically generate captions from images.
<img src="https://github.com/Adal-Glez/ComputerVision_ND_P2_Image-Captioning-Project/blob/master/image-captioning.png"/> 

# Model 
```python
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

...    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, dropout = 0.2, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        """
        super().__init__()
        """
    def forward(self, features, captions):
        
        batch_size = features.shape[0]
        captions_trimmed = captions[..., :-1]
        embed = self.embed(captions_trimmed)
        inputs = torch.cat([features.unsqueeze(1), embed], 1)
        lstm_out, self.hidden = self.lstm(inputs)
        outputs = self.fc(lstm_out)
        return outputs
```
The CNNEncoder class is a resnet50 with pretrained weights, whose parameters were clamped to avoid being trained and to do transfer learning. The last layer of the CNN was replaced by an embedding layer that will be trained.

The RNNDecoder  is made of an embedding layer, an LSTM, and a linear layer, which are properly used in the constructor, forward, and sample methods.


# Trainning
```python
## TODO #1: Select appropriate values for the Python variables below.
batch_size = 128          # batch size
vocab_threshold = 6        # minimum word count threshold
vocab_from_file = True    # if True, load existing vocab file
embed_size = 300           # dimensionality of image and word embeddings
hidden_size = 256          # number of features in hidden state of the RNN decoder
num_epochs = 3             # number of training epochs
save_every = 1             # determines frequency of saving model weights
print_every = 100          # determines window for printing average loss
log_file = 'training_log.txt'       # name of file with saved training loss and perplexity

# (Optional) TODO #2: Amend the image transform below.
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])
```
### CNN-RNN architecture
This NN consist of two parts: The CNN Encoder transforms a complex image into word embeddings using a pretrained ResNet-50 CNN for eature extraction. And the RNN Decoder learns the word embeddings of the descriptions and their connections in the form of a two layer LSTM network (with embedding layer prior to them).

I used the paper https://arxiv.org/pdf/1805.09137.pdf as reference and based on previous excercises I decided to fine tune the hyperparameter such as hidden layers, embed_size, also the minimum word count trashhold to 6 also try dropout at some point. Based on the lessons I decided to set batch at 128 and resulted in a was a good start.
### Transform
I decided to use the transformation defaults as a start since that same configuration was used at the preprocessing. Resize will match the image dimensions, Crop and HFlip gives partial images to search for overfitting, ToTensor and Normalization seemed reasonable.

### trainable parameters 
All the parameters of the RNNDecoder are trained from scratch. Whereas only the parameters in the last layer of the CNNEncoder are trained.

In this scenario I had to make a balance between time and resources each training was taking considerable amount of time and theres where pretrained models are helpful, focusing on the embeddings instead of the encoder and this is a reasonable step, and then depending on the results you can opt to fine tune deeper in the model.

### Optimizer
Adam Optimizer was a good choice for an optimizer. I considered alternatives that could be more robust to outliers in the data while training. However most recomendations goes to Adam with the Cross entroly loss function.

### Result 
Training ran for 3 epochs.
Loss evolved from 2.4274 at the beginning to 1.9303 at the end, which is good.

# Inference
```python
# TODO #1: Define a transform to pre-process the testing images.
transform_test = transforms.Compose([ 
                transforms.Resize(256),                          # smaller edge of image resized to 256
                transforms.RandomCrop(224),                      # get 224x224 crop from random location
                transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
                transforms.ToTensor(),                           # convert the PIL Image to a tensor
                transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                                     (0.229, 0.224, 0.225))])
```
The transform used to pre-process test images is congruent with the CNNEncoder and similar to the transform used to pre-process training images.

## Sample Method at Model.py
```python
def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        tokens = []
        for i in range(max_len):
            # Step through the sequence one element at a time.
            # after each step, hidden contains the hidden state.
            #out, hidden = lstm(i.view(1, 1, -1), hidden)

            lstm_output, states = self.lstm(inputs, states)
            out = self.fc(lstm_output)

            argmax = torch.argmax(out,dim=2)
            ind = argmax.item()
            tokens.append(argmax.item())

            #inputs = self.embed(argmax[1].long()).unsqueeze(1)
            if ind == 1:  # <end>
                break
            inputs = self.embed(argmax)
        return tokens
```
The sample method in RNNDecoder correctly leverages the RNN to generate predicted token indices.
This sample method verifies if the end of sentence appears. Hence it does not generate more words beyond the period.

The clean_sentence function passes the test in Step 4. And the sentence is reasonably clean, without <start> and <end>.
However, the first letter in the sentence is not in uppercase. And the end period has an extra space.
But these are minor issues that don't prevent us from passing this part of the rubric.

# Results
<img src="https://github.com/Adal-Glez/ComputerVision_ND_P2_Image-Captioning-Project/blob/master/captions.png"/> 

Thanks for reading
### Adalberto

# Interesting links:

If you want to investigate more on image captioning, the next logical step is to implement visual attention:
Show, Attend and Tell: Neural Image Caption - Generation with Visual Attention
http://proceedings.mlr.press/v37/xuc15.pdf

This tutorial is relevant to understand better what you have done with transfer learning:
TRANSFER LEARNING FOR COMPUTER VISION TUTORIAL
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

And of course, the official documentation of PyTorch is super important:
https://pytorch.org/

--

# Instructions  
1. Clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```

2. Setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```

3. Download some specific data from here: http://cocodataset.org/#download (described below)

* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)

4. The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order (`0_Dataset.ipynb, 1_Preliminaries.ipynb, 2_Training.ipynb, 3_Inference.ipynb`).
