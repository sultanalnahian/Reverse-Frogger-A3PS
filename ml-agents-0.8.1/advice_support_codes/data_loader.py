import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from advice_generator.build_vocab_v2 import Vocabulary
#from build_vocab_v2 import Vocabulary
#from advice_generator.frogger_dataset import FroggerDataset
from advice_generator.frogger_dataset_preprocessed import FroggerDataset
import cv2



def create_rationalization_matrix(rationalization, vocab):
    max_rationalization_len = 40
    rationalization_matrix = []
    sequence_lenght_arr = []
    #sent = rationalization
    sent = rationalization.replace("'","")
    sent = sent.strip()
    words = sent.lower().split(' ')
    
    ration_sent = np.zeros([ max_rationalization_len ], dtype=np.int32)
    ration_sent[0] = 1
    idx = 1
    
    for k, word in enumerate(words):
        if idx == max_rationalization_len:
            break
        if word in vocab.word2idx:
            ration_sent[idx] = vocab.word2idx[word]
        else:
            ration_sent[idx] = 3
        idx +=1
        
    if idx < max_rationalization_len:
        ration_sent[idx] = 2
        idx += 1
    rationalization_matrix.append(ration_sent)
    sequence_lenght_arr.append(idx)
        
    rationalization_matrix = np.array(rationalization_matrix)
    return rationalization_matrix


class FroggerDataLoader(data.Dataset):
    def __init__(self,vocab,rationalizations, images, cur_image_dir,action): 
        
        self.vocab = vocab
        self.rationalization = rationalizations
        #print("self.advices: ",(self.rationalization))
       
        self.images = images
        #print("self.images: ",(self.images))
        self.image_dir = cur_image_dir
        self.action = action
        
        
    def __getitem__(self,index):
        
        image_name = self.images[index]
        action = self.action[index]
        advice = self.rationalization[index]
        #print("advice: ",advice, "image: ",image_name, "action: ",action)
        
        img_path = os.path.join(self.image_dir, image_name)
        img_arr = cv2.imread(img_path)        
        resized_img = cv2.resize(img_arr, (224,224))      
        x = np.reshape(resized_img, (3,224,224))        
        image_arr = torch.Tensor(x)
        
        advice_ids = create_rationalization_matrix(advice, self.vocab)
        #print("advice_testing: ", advice_ids)
        advice_text = torch.tensor(advice_ids)
        advice_text = advice_text.type(torch.LongTensor)
        #print("advice_text_shape: ",advice_text.shape)
        advice_text = torch.reshape(advice_text,(-1,))
        #advice_text = torch.from_numpy(advice_ids)
        
        
        #action_arr = torch.zeros(5)
        #action_arr[action] = 1
        action_arr = torch.tensor(action)
        #action_arr = action_arr.type(torch.LongTensor)
        
        return advice_text, image_arr, action_arr
    
    def __len__(self):
        return len(self.rationalization)
    
def get_loader(vocab, advices, images, cur_image_dir, act, batch_size,transform, shuffle, num_workers):
    frogger = FroggerDataLoader(vocab, advices, images, cur_image_dir, act)

    data_loader = torch.utils.data.DataLoader(dataset=frogger, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader
        