import numpy as np
from xlrd import open_workbook
from collections import Counter
import nltk
import re
from advice_generator.build_vocab_v2 import Vocabulary
from math import floor
import os
from nltk.corpus import stopwords
import pickle as pkl

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

class FroggerDataset():
    def __init__(self, data_file, vocab):
        self.data_file = data_file
        self.vocab = vocab
    
    def create_rationalization_matrix(self, rationalizations, vocab):
        rationalization_matrix = []
        sequence_lenght_arr = []
        for sent in rationalizations:
            sent = sent.replace("'","")
            sent = sent.strip()
            words = sent.lower().split(' ')
            max_rationalization_len = len(words)
            max_rationalization_len = max_rationalization_len + 2
            ration_sent = np.zeros([ max_rationalization_len ], dtype=np.int32)
            ration_sent[0] = 1
        
            for k, word in enumerate(words):
                if word in vocab.word2idx:
                    ration_sent[k+1] = vocab.word2idx[word]
                else:
                    ration_sent[k+1] = 3
            
            ration_sent[max_rationalization_len-1] = 2
            rationalization_matrix.append(ration_sent)
            sequence_lenght_arr.append(max_rationalization_len+2)

        return rationalization_matrix, sequence_lenght_arr

    def preprocessing(advice):
    #rationalization = []
        line = advice.lower()
        line = line.replace('\'','')
        line = line.replace('.','')
        line = line.replace(',','')
        line = line.replace('``','')
        line = line.replace('"','')
        line = line.replace('\\n','')
        line = line.replace(':','')
        line = line.replace('(','')
        line = line.replace(')','')
        
        return line
        
    def load_data(self):
        data_file = self.data_file
        vocab = self.vocab
        wb = open_workbook(data_file)
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
            counter = Counter()
            for row in range(1, number_of_rows):
                values = []
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
                    word_token= [w for w in tokens if not w in stop_words]
                    #line = line.lower()
                    good_rationalizations.append(line)
                    #line = re.sub('[^a-z\ ]+', " ", line)
                    line = re.sub(r'[^\w\s]',' ', line)
                    #word_token= [w for w in tokens if not w in stop_words]
                    words = line.split()
                    
                    length = len(word_token)
                    if length == 0:
                        action = sheet.cell(row,2).value
                        action = action.lower()
                        word_token.append(action)
                        print("short sentence: ", word_token)
                    lengths.append(length)
                    if length>max_length:
                        max_length = length
                    new_id_list = []
                    new_id_list.append(vocab.word2idx['<start>'])
                    for index,word in enumerate(word_token):   
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
            rationalization_tokens = [np.array(xi) for xi in rationalization_tokens]
			
        #good_ids_split = int(floor((90.0/100)*len(good_ids)))
        with open("sentence_len.pkl", "wb") as output_file:
            pkl.dump( lengths,output_file)
        with open("sent_len_no_unk.pkl", "wb") as output_file:
            pkl.dump(lengths_unk,output_file)

        print("max_length: ",max_length)
        split = int(floor((95.0/100)*len(rationalizations)))
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
        #print("te_contents: ",te)
        #training_rationalizations = good_rationalizations[tr]
        #testing_rationalizations = good_rationalizations[te]
        training_rationalizations = rationalization_tokens[tr]
        print("train rationalizations_tokens: ",len(training_rationalizations))
        testing_rationalizations = rationalization_tokens[te]
        print("test rationalizations_tokens: ",len(testing_rationalizations))
        
        return tr_good_ids, te_good_ids, tr_indices, te_indices, training_rationalizations, testing_rationalizations

    def load_images(self, current_image_dir,  tr_good_ids, te_good_ids, tr_indices):
        current_images = os.listdir(current_image_dir)
        current_images = sorted(current_images ,key = numericalSort)
		
		#print(good_ids[tr_indices[0]]
        
        #next_images = os.listdir(next_image_dir)
        #next_images = sorted(next_images ,key = numericalSort)
        num_images = len(current_images)
        print("number of images: ",num_images)
        
        cur_training_images = []
        #next_training_images = []
        cur_test_images = []
        #next_test_images = []
        for i, file in enumerate(current_images):
            if i in tr_good_ids:
                cur_training_images.append(file)
                #if good_ids[tr_indices[0]]<=i and i<=good_ids[tr_indices[1]]:
                    #cur_training_images.append(file)
                    #next_training_images.append(next_images[i])
            elif i in te_good_ids:
                cur_test_images.append(file)
                    #next_test_images.append(next_images[i])
        #print("number of test images: ",len(cur_test_images))       
        return cur_training_images, cur_test_images        
