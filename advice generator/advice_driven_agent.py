import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import tensorflow as tf
import json
import pickle
import copy
import numpy as np
from xlrd import open_workbook
from collections import Counter
from torchvision import transforms
from data_loader_advice_driven_train import FroggerDataLoader, get_loader, create_rationalization_matrix
import nltk
import re
import pickle
from build_vocab_v2 import Vocabulary
from math import floor
import cv2
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from PIL import Image
from nltk.corpus import stopwords
from advice_models import EncoderCNN, DecoderRNN, Advice_Encoder, Action_Decoder

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


actions_map = dict()
actions_map['Down'] = 0
actions_map['Left'] = 1
actions_map['Right'] = 2
actions_map['Up'] = 3
actions_map['Wait'] = 4

def calculate_stat(action):
    action_stat = dict()
    act_down = 0
    act_left = 0
    act_right = 0
    act_up = 0
    act_wait = 0

    #i = 0
    length = len(action)
    #action = action.unsqueeze(0)

    for i in range(length):
        j = action[i].item()
        if (action[i].item() == 0):
            act_down +=1 
        elif (action[i].item() == 1):
            act_left +=1
        elif (action[i].item() == 2):
            act_right +=1
        elif (action[i].item() == 3):
            act_up +=1
        elif (action[i].item() == 4):
            act_wait +=1

    action_stat['Down'] = act_down
    action_stat['Left'] = act_left
    action_stat['Right'] = act_right
    action_stat['Up'] = act_up
    action_stat['Wait'] = act_wait

    return action_stat



def weight_gloVe(target_vocab):
    glove = pickle.load(open(f'glove/6B_300_words_emb.pkl', 'rb'))

    matrix_len = len(target_vocab.word2idx)
    weights_matrix = np.zeros((matrix_len, 300))
    words_found = 0
    emb_dim = 300
    
    for i in range(matrix_len):
        word = target_vocab.idx2word[i]
        try: 
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
    
    print("words_found: ", words_found)
    
    return weights_matrix

def load_data_preprocess(data_file, vocab_path):
        #data_file = data_file
        #vocab = self.vocab
    wb = open_workbook(data_file)
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    stop_words = set(stopwords.words('english'))

        #read the rationalizations from the excel file and create a list of training/testing rationalizations. 
    for sheet in wb.sheets():
        number_of_rows = sheet.nrows
        number_of_columns = sheet.ncols
        rationalizations = []
        rationalization_tokens = []
        items = []
        rows = []
        lengths = []
        lengths_unk = []
            
        max_length = 0
            
        bad_worker_ids = ['A2CNSIECB9UP05','A23782O23HSPLA','A2F9ZBSR6AXXND','A3GI86L18Z71XY','AIXTI8PKSX1D2','A2QWHXMFQI18GQ','A3SB7QYI84HYJT',
    'A2Q2A7AB6MMFLI','A2P1KI42CJVNIA','A1IJXPKZTJV809','A2WZ0RZMKQ2WGJ','A3EKETMVGU2PM9','A1OCEC1TBE3CWA','AE1RYK54MH11G','A2ADEPVGNNXNPA',
    'A15QGLWS8CNJFU','A18O3DEA5Z4MJD','AAAL4RENVAPML','A3TZBZ92CQKQLG','ABO9F0JD9NN54','A8F6JFG0WSELT','ARN9ET3E608LJ','A2TCYNRAZWK8CC',
    'A32BK0E1IPDUAF','ANNV3E6CIVCW4','AXMQBHHU22TSP','AKATSYE8XLYNL','A355PGLV2ID2SX','A55CXM7QR7R0N','A111ZFNLXK1TCO']
            
        good_ids = []
        good_rationalizations = []
        actions = []
        counter = Counter()
        for row in range(1, number_of_rows):
            values = []
            word_list = []
            worker_id = sheet.cell(row,0).value
            if worker_id not in bad_worker_ids:
                good_ids.append(row-1)
                line = sheet.cell(row,4).value
                line = line.lower()
                line = line.replace('\'','')
                line = line.replace('.','')
                line = line.replace(',','')
                line = line.replace('``','')
                line = line.replace('"','')
                line = line.replace('\\n','')
                line = line.replace(':','')
                line = line.replace('(','')
                line = line.replace(')','')
                    #line = self.preprocessing(line)
                tokens = nltk.tokenize.word_tokenize(line)
                sentence = ""
                #word_token= [w for w in tokens if not w in stop_words]
                for w in tokens:
                    if not w in stop_words:
                        sentence = sentence + " " + w 
                        word_list.append(w)

                #good_rationalizations.append(sentence)
                    #line = re.sub('[^a-z\ ]+', " ", line)
                line = re.sub(r'[^\w\s]',' ', line)
                words = line.split()
                _action = sheet.cell(row,2).value
                actions.append(actions_map[_action])
                    
                length = len(word_list)
                if length == 0:
                    #action = sheet.cell(row,2).value                        
                    action = _action.lower()
                    sentence = sentence + " " + action
                    #word_token.append(action)
                    print("short sentence: ", sentence)
                good_rationalizations.append(sentence)
                lengths.append(length)
                if length>max_length:
                    max_length = length
                new_id_list = []
                new_id_list.append(vocab.word2idx['<start>'])
                for index,word in enumerate(word_list):
                    if word in vocab.word2idx:
                        id_word = vocab.word2idx[word]
                        new_id_list.append(id_word)
                        
                        #else:
                            #word_token[index] = vocab.word2idx['<unk>']
                length_unk = len(new_id_list)
                lengths_unk.append(length_unk)
                rationalizations.append(words)
                rationalization_tokens.append(new_id_list)
        rationalizations=[np.array(xi) for xi in rationalizations]
        #rationalization_tokens = [np.array(xi) for xi in rationalization_tokens]
			
        #good_ids_split = int(floor((90.0/100)*len(good_ids)))
    with open("sentence_len.pkl", "wb") as output_file:
        pickle.dump( lengths,output_file)
    with open("sent_len_no_unk.pkl", "wb") as output_file:
        pickle.dump(lengths_unk,output_file)

    print("max_length: ",max_length)
    split = int(floor((90.0/100)*len(rationalizations)))
    print("good_rationalizations: ",len(good_rationalizations))
        #print("good_rationalizations: ",good_rationalizations)		
    print("rationalizations_tokens: ",len(rationalization_tokens))
        #print("rationalizations: ",rationalizations)        
    tr = slice(0,split)
    tr_good_ids = good_ids[tr]
    print("tr_good_ids: ",len(tr_good_ids))
        		
    tr_indices = [0,split-1]
        #print("tr_contents: ",tr)
    te_indices = [split,len(rationalizations)-1]

    te = slice(split,len(rationalizations))
    te_good_ids = good_ids[te]
    print("te_good_ids: ",len(te_good_ids))		

    training_rationalizations = good_rationalizations[tr]
    #print("train rationalizations_tokens: ",len(training_rationalizations))
    testing_rationalizations = good_rationalizations[te]
    training_actions = actions[tr]
    testing_actions = actions[te]
    #print("test rationalizations_tokens: ",len(testing_rationalizations))
        
    return tr_good_ids, te_good_ids, tr_indices, te_indices, training_rationalizations, testing_rationalizations, training_actions, testing_actions, vocab

def load_data(data_file, vocab_path):
    wb = open_workbook(data_file)
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    #vocab = vocab
    #read the rationalizations from the excel file and create a list of training/testing rationalizations. 
    for sheet in wb.sheets():
        number_of_rows = sheet.nrows
        number_of_columns = sheet.ncols
        rationalizations = []
        items = []
        rows = []
        lengths = []
        max_length = 0
        
        bad_worker_ids = ['A2CNSIECB9UP05','A23782O23HSPLA','A2F9ZBSR6AXXND','A3GI86L18Z71XY','AIXTI8PKSX1D2','A2QWHXMFQI18GQ','A3SB7QYI84HYJT',
'A2Q2A7AB6MMFLI','A2P1KI42CJVNIA','A1IJXPKZTJV809','A2WZ0RZMKQ2WGJ','A3EKETMVGU2PM9','A1OCEC1TBE3CWA','AE1RYK54MH11G','A2ADEPVGNNXNPA',
'A15QGLWS8CNJFU','A18O3DEA5Z4MJD','AAAL4RENVAPML','A3TZBZ92CQKQLG','ABO9F0JD9NN54','A8F6JFG0WSELT','ARN9ET3E608LJ','A2TCYNRAZWK8CC',
'A32BK0E1IPDUAF','ANNV3E6CIVCW4','AXMQBHHU22TSP','AKATSYE8XLYNL','A355PGLV2ID2SX','A55CXM7QR7R0N','A111ZFNLXK1TCO']
        
        good_ids = []
        good_rationalizations = []
        actions = []
        counter = Counter()
        for row in range(1, number_of_rows):
            values = []
            worker_id = sheet.cell(row,0).value
            if worker_id not in bad_worker_ids:
                good_ids.append(row-1)
                line = sheet.cell(row,4).value
                tokens = nltk.tokenize.word_tokenize(line.lower())
                # if tokens!=[]:
                _action = sheet.cell(row,2).value
                actions.append(actions_map[_action])
                line = line.lower()
                good_rationalizations.append(line)
                #line = re.sub('[^a-z\ ]+', " ", line)
                line = re.sub(r'[^\w\s]',' ', line)
                words = line.split()
                length = len(tokens)
                lengths.append(length)
                if length>max_length:
                    max_length = length
                for index,word in enumerate(tokens): 
                    if word in vocab.word2idx:
                        tokens[index] = vocab.word2idx[word]
                    else:
                        tokens[index] = vocab.word2idx['<unk>']
                    
                rationalizations.append(words)
        rationalizations=[np.array(xi) for xi in rationalizations]

    split = int(floor((90.0/100)*len(rationalizations)))
    
    tr = slice(0,split)
    tr_good_ids = good_ids[tr]
    print("tr_good_ids: ",type(tr_good_ids))
    tr_indices = [0,split-1]
    te_indices = [split,len(rationalizations)-1]
    te = slice(split,len(rationalizations))
    te_good_ids = good_ids[te]
    print("te_good_ids: ",len(te_good_ids))
    training_rationalizations = good_rationalizations[tr]
    testing_rationalizations = good_rationalizations[te]
    training_actions = actions[tr]
    testing_actions = actions[te]
   
    training_rationalizations_text = good_rationalizations[tr]
    testing_rationalizations_text = good_rationalizations[te]   
 
    
    return tr_good_ids,te_good_ids, tr_indices, te_indices, training_rationalizations, testing_rationalizations, training_actions, testing_actions, vocab
    

def load_images(current_image_dir, next_image_dir, tr_good_ids, tr_indices):
    current_images = os.listdir(current_image_dir)
    current_images = sorted(current_images ,key = numericalSort)
    num_images = len(current_images)
    #print("num_next_images: ",num_images)
    next_images = os.listdir(next_image_dir)
    next_images = sorted(next_images ,key = numericalSort)
    num_next_images = len(next_images)
    print("_next_images: ",type(next_images))
    
    cur_training_images = []
    next_training_images = []
    cur_test_images = []
    next_test_images = []
    for i, file in enumerate(current_images):
        if i in tr_good_ids:
            cur_training_images.append(file)
            next_training_images.append(next_images[i])
        else:
            cur_test_images.append(file)
            next_test_images.append(next_images[i])
    
    
    return cur_training_images, next_training_images, cur_test_images, next_test_images

advice_dim = 512
image_size = 224
batch_size = 8
width = 320
height = 320
depth = 3
#image_embedding_size = 512
image_embedding_size = 2048
hidden_size = 512
sentence_embedding_size = 512
word_embedding_size = 512
vocab_size = 539

num_output = 5
drop_out_rate = 0.5
learning_rate = 0.001
max_epoch = 101
dim_att = 1024

def train(cur_image_dir, next_image_dir):
    
    train_transform = transforms.Compose([
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])
    
    val_transform = transforms.Compose([
    transforms.Resize(image_size, interpolation=Image.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])
    
    tr_good_ids,te_good_ids, training_indices, testing_indices, training_rationalizations, testing_rationalizations, trn_act, tst_act, vocab = load_data_preprocess("Turk_Master_File_sorted.xlsx", 'data/vocab_frogger_preprocessed.pkl')
    cur_training_images, next_training_images, cur_test_images, next_test_images = load_images(current_image_dir, next_image_dir, tr_good_ids, training_indices)
    #print("training_rationalizations: ", training_rationalizations)

    train_data_loader = get_loader(vocab,training_rationalizations, cur_training_images,
                                   current_image_dir, trn_act, batch_size, train_transform,
                                   shuffle=True, num_workers=0)
    
    
    
    val_data_loader = get_loader(vocab,testing_rationalizations, cur_test_images,
                                   current_image_dir, tst_act, batch_size, val_transform,
                                   shuffle=True, num_workers=0)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(train_data_loader.dataset.vocab)
    weights_= weight_gloVe(vocab)
    weights_ = torch.from_numpy(weights_).float()
    vocab_size, word_embedding_size =  weights_.shape
    image_encoder = EncoderCNN(image_embedding_size).to(device)
    #word_emb = WordEmbedding(vocab_size,embedding_dim, 0.0)
    #word_emb.init_embedding(weights_)
    #advice_encoder = Sentence_Encoder(embedding_dim,dim_hidden,1,0.0)
    advice_encoder = Advice_Encoder(vocab_size,word_embedding_size,hidden_size,1,0.0).to(device)
    advice_encoder.init_word_embedding(weights_)
    decoded_action = Action_Decoder(image_embedding_size,sentence_embedding_size,num_output,0.2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    params = list(image_encoder.parameters())+ list(advice_encoder.parameters())+list(decoded_action.parameters())
    optimizer = torch.optim.Adamax(params, lr=0.001, weight_decay=1e-5)
    
    train_step = len(train_data_loader)
    val_step = len(val_data_loader)
    
    train_losses = []
    val_losses = []
    
    file1 = open("models/advice_driven_agent_loss_100.txt","a+")
    
    #max_epoch = 2
    for epoch in range (0,max_epoch):
        print("Epoch", epoch, "is starting....")
        train_loss = 0.0
        
        image_encoder.train()
        #word_emb.train()
        advice_encoder.train()
        decoded_action.train()
        
        for i, (adv,img,act,img_name) in enumerate(train_data_loader):
            img = img.to(device)
            adv = adv.to(device)
            act = act.to(device)
            encoded_image= image_encoder(img)
            embedded_adv,hidden = advice_encoder(adv)
            predicted_action = decoded_action(encoded_image,hidden)
                        
            loss = criterion(predicted_action,act) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Testing the models    
        if (epoch% 5 == 0):
            image_encoder.eval()
            advice_encoder.eval() 
            decoded_action.eval()
        
            total_val_loss = 0.0
            true_action_score = 0
            pred_act_score =0
            true_act_list = {}
            predict_act_list = {}

            print("Validation process is starting....")
            
            with torch.no_grad():
                for j, (adv,img,act,img_name) in enumerate(val_data_loader):
                    img = img.to(device)
                    adv = adv.to(device)
                    act = act.to(device)
                    #true_act_stat = calculate_stat(act)
                    encoded_image= image_encoder(img)
                    embedded_adv,hidden = advice_encoder(adv)
                    predicted_action = decoded_action(encoded_image, hidden)
                    #true_act_list.append(act)
                    
                    val_loss = criterion(predicted_action,act) 
                    #print("real action: ",act)
                    true_action_score += act.size(0)
                    pred_act= torch.max(predicted_action,1)[1].data
                    #predict_act_list.append(pred_act)
                    #predicted_act_stat = calculate_stat(pred_act)
                                        
                    pred_act_score +=(pred_act==act).sum().item()
                                                                
                    val_accuracy = 100 * pred_act_score / true_action_score
                    img_test = img_name
                
                    total_val_loss += val_loss.item()                
                
                    
                #print("true_act_stat: ", true_act_stat)
                #print("predicted_act_stat: ", predicted_act_stat)
                val_avg_loss = total_val_loss/ val_step
                print("Avg Validation loss after epoch: ",epoch, "is: ",val_avg_loss )
                file1.write("for epoch:%d, Average_Validation_loss:%.3f\n"% (epoch,val_avg_loss))
            print("Test accuracy score after epoch: ",epoch, "is: ",val_accuracy)
            
        train_loss_avg =  train_loss/train_step
        
        print("Average_Train_loss: ",train_loss_avg)
        #print("Average_Validation_loss: ", val_loss_avg)
        train_losses.append(train_loss_avg)   
        val_losses.append(val_avg_loss)
        #write mode 
        file1.write("Average_Train_loss:%.3f \n"% (train_loss_avg)) 
         
        
        if (epoch% 10 == 0):
            
            print("\nSaving the models")
            torch.save(image_encoder.state_dict(), os.path.join('models/', 'image_encoder_pre_we300_im2048-%d.pth' % epoch))
            torch.save(advice_encoder.state_dict(), os.path.join('models/', 'advice_encoder_pre_we300_im2048-%d.pth' % epoch))
            torch.save(decoded_action.state_dict(), os.path.join('models/', 'decoder_pre_we300_im2048-%d.pth' % epoch))        
    file1.close()   

def test_agent(encoded_img, advice_text) :
    vocab_path = 'data/vocab_frogger.pkl'
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    advice_ids = create_rationalization_matrix(advice, vocab)
    #print("advice_testing: ", advice_ids)
    advice_text = torch.tensor(advice_ids)
    advice_text = advice_text.type(torch.LongTensor)
    #print("advice_text_shape: ",advice_text.shape)
    advice_text = torch.reshape(advice_text,(-1,))
    
    word_emb = torch.load("word_emb-100.pth")
    sentence_emb = torch.load("sentence_encoder-100.pth")
    action_decode = torch.load("decoder-100.pth")
    
    words = word_emb(advice_text)
    sentences = sentence_emb(words)
    actions = action_decode (encoded_img, sentences)
    
    pred_act= torch.max(actions,1)[1].data
    
    return pred_act

current_image_dir = 'data/All_Images/Current_State/'
next_image_dir = 'data/All_Images/Next_State/'
train(current_image_dir, next_image_dir)
    
    
    
    
    
