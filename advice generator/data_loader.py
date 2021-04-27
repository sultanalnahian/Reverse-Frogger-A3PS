import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab_v2 import Vocabulary
from frogger_dataset import FroggerDataset
import cv2
from advice_models import EncoderCNN, DecoderRNN

image_size = 100
advice_rnn_hidden_size = 512
img_embed_size = 512
batch_size = 8
#depth = 3
#image_embedding_size = 512
sent_emb_size = 512

num_output = 5
learning_rate = 0.001
max_epoch = 101
n_layer = 1
word_embed_size = 300
vocab_size = 1104

# advice_gen_img_encoder = EncoderCNN(img_embed_size)
# advice_gen_decoder = DecoderRNN(word_embed_size, img_embed_size, advice_rnn_hidden_size, vocab_size)

# if torch.cuda.is_available():
#     advice_gen_img_encoder.cuda()
#     advice_gen_decoder.cuda()

# advice_gen_img_encoder.load_state_dict(torch.load('./models/train_advice_agent/encoder_gray-380.pth'))
# advice_gen_decoder.load_state_dict(torch.load('./models/train_advice_agent/decoder_gray-380.pth'))



def create_rationalization_matrix(rationalization, vocab):
    max_rationalization_len = 18
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
        #else:
            #ration_sent[idx] = 3
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
        self.height = 100
        self.width = 100
        self.images = images
        #print("self.images: ",(self.images))
        self.image_dir = cur_image_dir
        self.action = action
        
        
    def __getitem__(self,index):
        
        image_name = self.images[index]
        action = self.action[index]
        advice = self.rationalization[index]
        
        img_path = os.path.join(self.image_dir, image_name)
        img_arr = cv2.imread(img_path)        
        resized_img = cv2.resize(img_arr, (self.height,self.width))      
        x = np.reshape(resized_img, (3,self.height,self.width))        
        image_arr = torch.Tensor(x)

        #image_feature = advice_gen_img_encoder(image_arr)
        #advice_ids = advice_gen_decoder.inference(image_feature)
        #advice_ids = torch.tensor(advice_ids).to(device)
        #advice_ids = advice_ids.view(1,-1)
        
        advice_ids = create_rationalization_matrix(advice, self.vocab)
        advice_text = torch.tensor(advice_ids)
        advice_text = advice_text.type(torch.LongTensor)
        advice_text = torch.reshape(advice_text,(-1,))

        action_arr = torch.tensor(action)
        #action_arr = action_arr.type(torch.LongTensor)
        
        return advice_text, image_arr, action_arr, image_name 
    
    def __len__(self):
        return len(self.rationalization)
    
def get_loader(vocab, advices, images, cur_image_dir, act, batch_size,transform, shuffle, num_workers):
    frogger = FroggerDataLoader(vocab, advices, images, cur_image_dir, act)

    data_loader = torch.utils.data.DataLoader(dataset=frogger, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader
        