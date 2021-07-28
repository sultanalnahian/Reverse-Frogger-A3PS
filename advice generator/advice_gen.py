import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from build_vocab_v2 import Vocabulary
from torchvision import transforms
from data_loader_advice_gen import get_loader
#from frogger_dataset import FroggerDataset
import pickle
from PIL import Image
from frogger_dataset_preprocessed import FroggerDataset
import os
import torchvision.models as models
import numpy as np
import sys
import json
import cv2
from advice_models import image_EncoderCNN, DecoderRNN, EncoderCNN

# num_epochs = 101
#total_step = 50

#hidden_size = 1024
# hidden_size =512
# embed_size = 512
# image_size = 224
# batch_size =8

with open("data/vocab_frogger_preprocessed.pkl", 'rb') as f:
    in_vocab = pickle.load(f)
with open("data/input_vocab.pkl", 'rb') as f:
    image_vocab = pickle.load(f)
        
current_image_dir = 'data/All_Images/Current_State/'

def get_length(source_list):
    length = len(source_list)
    try:
        length = list(source_list).index(0)
    except:
        length = len(source_list)
    return length

def get_sentences(word_ids):
    batch = len(word_ids[0])
    sent_len = len(word_ids)
    sentences = []
    for col in range(batch):
        sentence = ""
        for row in range(sent_len):
            word = in_vocab.idx2word[int(word_ids[row][col])]
            if word != '<end>' and word != '<pad>' and word != '<start>':
                sentence = sentence + " " + word
        
        sentence = sentence.strip()
        sentences.append(sentence)
    return sentences

def weight_gloVe(target_vocab):
    # words = pickle.load(open(f'glove.6B/6B.300_words.pkl', 'rb'))
    # word2idx = pickle.load(open(f'glove.6B/6B.300_idx.pkl', 'rb'))
    # vectors = bcolz.open(f'glove.6B/6B.300.dat')[:]
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

def main(args):
    train_transform = transforms.Compose([
        transforms.RandomCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])

    val_transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])
        
    frogger_dataset_ob = FroggerDataset('Turk_Master_File_sorted.xlsx', in_vocab)
    tr_good_ids, te_good_ids, tr_indices, te_indices, training_rationalizations, testing_rationalizations = frogger_dataset_ob.load_data()
    cur_training_images, cur_test_images = frogger_dataset_ob.load_images(current_image_dir, tr_good_ids, te_good_ids,tr_indices)
    print("numbers of Train images: ",len(cur_training_images))
    print("numbers of test images: ",len(cur_test_images))
        
    train_data_loader = get_loader(in_vocab,training_rationalizations,cur_training_images,
                                current_image_dir, args.batch_size, train_transform,
                                shuffle=True, num_workers=0)
    val_data_loader = get_loader(in_vocab,testing_rationalizations,cur_test_images,
                                current_image_dir,args.batch_size, val_transform,
                                shuffle=True, num_workers=0)

    # val_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(train_data_loader.dataset.vocab)
    pretrained_word_weight = weight_gloVe(in_vocab)
    pretrained_word_weight = torch.from_numpy(pretrained_word_weight).float()
    pretrained_word_weight = pretrained_word_weight.to(device)
    #encoder = image_EncoderCNN(args.img_feature_size).to(device)
    encoder= EncoderCNN(args.img_feature_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.img_feature_size, args.hidden_size, vocab_size).to(device)
    decoder.init_word_embedding(pretrained_word_weight)
        
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=1e-5)

    total_train_step = len(train_data_loader)
    print("numbers of train images: ",total_train_step)
    total_val_step = len(val_data_loader)
    print("numbers of val images: ",total_val_step)



    #if os.path.exists('models/decoder-90.pth'):
        #decoder.load_state_dict(torch.load('models/decoder-90.pth'))
        #print('decoder Model loaded')
    #if os.path.exists('models/encoder-90.pth'):
        #encoder.load_state_dict(torch.load('models/encoder-90.pth'))       
        #print('encoder Model loaded')
        
    #num_epochs = 1
    train_losses = []
    val_losses = []
    for epoch in range(0, args.num_epochs):  
        results = []      
        inference_results = []
        total_train_loss = 0
                       
        # set decoder and encoder into train mode
        encoder.train()
        decoder.train()
        for i_step in range(0, total_train_step):
            # zero the gradients
            decoder.zero_grad()
            encoder.zero_grad()
    
            
            # Obtain the batch.
            img_name, images, captions = next(iter(train_data_loader))
            
            # make the captions for targets and teacher forcer
            captions_target = captions[:, 1:].to(device)
            captions_train = captions[:, :captions.shape[1]].to(device)

            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(device)
            
            # Pass the inputs through the CNN-RNN model.
            features = encoder(images)
            outputs = decoder(features, captions_train)
            
            # Calculate the batch loss
            loss = 0.0
            for sj, output_result in enumerate(zip(outputs, captions_target)):
                length = get_length(output_result[1])
                x = output_result[0]
                y = output_result[1]
                loss += criterion(x[:length,], y[:length])
            #loss = criterion(outputs.view(-1, vocab_size), captions_target.contiguous().view(-1))
            
            # Backward pass
            # optimizer.zero_grad()
            loss.backward()
            
            # Update the parameters in the optimizer
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss/total_train_step
        train_losses.append(avg_train_loss)
        # - - - Validate - - -
        # turn the evaluation mode on
        with torch.no_grad():
            # set the evaluation mode
            encoder.eval()
            decoder.eval()
            total_val_loss = 0
            for val_step in range(0, total_val_step):
                # get the validation images and captions
                img_name, val_images, val_captions = next(iter(val_data_loader))

                # define the captions
                captions_target = val_captions[:, 1:].to(device)
                captions_train = val_captions[:, :val_captions.shape[1]-1].to(device)

                # Move batch of images and captions to GPU if CUDA is available.
                val_images = val_images.to(device)

                # Pass the inputs through the CNN-RNN model.
                features = encoder(val_images)
                outputs = decoder(features, captions_train)
                output_ids = decoder.inference(features)
                cap_predicted = outputs.cpu().numpy()
                predicted_sentences = get_sentences(output_ids)
                
                for i in range (0, args.batch_size):
                    caption_str = ""
                    caption_grndtrth = ""
                    for j in range (0,len(cap_predicted[i])):
                        out_str = np.argmax(cap_predicted[i][j])
                        #print("out_str: ",out_str)
                        out_ = int(out_str)
                        #out_ = str(out_)
                        out_cap = in_vocab.idx2word[out_str]
                        caption_str = caption_str + " " + out_cap
                        
                    for k in range (0,len(outputs[i])):
                        grnd_str = captions_target[i][k]
                        #print("grnd_str: ", grnd_str)
                        grnd_ = int(grnd_str)
                        #print("grnd_: ", grnd_)
                        #grnd_ = str(grnd_)
                        grnd_cap = in_vocab.idx2word[grnd_]
                        caption_grndtrth = caption_grndtrth + " " + grnd_cap
                    
                    image_name = img_name[i]
                    
                    
                    results.append({u'image name': image_name, u'generated caption': caption_str, u'ground_truth_cap': caption_grndtrth})
                    inference_results.append({u'image name': image_name, u'generated caption': predicted_sentences[i], u'ground_truth_cap': caption_grndtrth})

                    #print("generated caption: ", caption_str, " ground_truth_cap: ", caption_grndtrth)
                    #" ground_ans: ", true_ans)
                        

                # Calculate the batch loss.
                val_loss = criterion(outputs.view(-1, vocab_size), captions_target.contiguous().view(-1))
                total_val_loss += val_loss.item()
            
            avg_val_loss = total_val_loss/total_val_step
            val_losses.append(avg_val_loss)
            
            # save the losses
            
            
            # Get training statistics.
        stats = 'Epoch [%d/%d], Training Loss: %.4f, Val Loss: %.4f' % (epoch, args.num_epochs, avg_train_loss, avg_val_loss)
            
        # Print training statistics (on same line).
        print('\r' + stats)
        #sys.stdout.flush()
                
        # Save the weights.
        if epoch % 20 == 0:
            print("\nSaving the model")
            torch.save(decoder.state_dict(), os.path.join('models/', 'decoder_prep95_dl-%d.pth' % epoch))
            torch.save(encoder.state_dict(), os.path.join('models/', 'encoder_prep95_dl-%d.pth' % epoch))
            my_advice = list(results)
            inference_advice = list(inference_results)
            json.dump(my_advice,open('results/advices_img_prep95-epoch-%d.json' % epoch,'w'))  
            json.dump(inference_advice,open('results/inference_advice_img_prep95-epoch-%d.json' % epoch,'w'))  
    
    np.save('results/train_losses', np.array(train_losses))
    np.save('results/val_losses', np.array(val_losses))       
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', type=int, default=100 ,
                        help='size for input images')
    parser.add_argument('--img_feature_size', type=int , default=512,
                        help='dimension of image feature')
    parser.add_argument('--embed_size', type=int , default=300 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--max_words', type=int , default=40,
                        help='maximum number of words in a sentence')
    parser.add_argument('--num_epochs', type=int, default=301)
    parser.add_argument('--batch_size', type=int, default=8)

    
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    args = parser.parse_args()
    print(args)
    main(args)

# def test_adv_gen(img):
#     img_arr = cv2.imread(img)
#     #print("image_arr after cv2: ", img_arr.shape)
#     resized_img = cv2.resize(img_arr, (224,224))
#     x = np.reshape(resized_img, (3,224,224))
#     image_arr = torch.Tensor(x)
    
#     encoder = EncoderCNN(embed_size)
#     decoder = DecoderRNN( embed_size,hidden_size, vocab_size)
    
#     img_path = "models/encoder_drpout_v2-100.pth"
#     enocder.load_state_dict(torch.load(img_path))
#     advice_path = "models/decoder_drpout_v2-100.pth"
#     #model = DecoderRNN(image_embedding_size,dim_hidden,vocab_size)
#     decoder.load_state_dict(torch.load(advice_path))
    
#     encoded_img = encoder(image_arr)
#     decoded_adv = decoder(encoded_img)
#     decoded_adv = decoded_adv.cpu().numpy()
    
#     caption_str = ""
#     adv_list = []
#     for j in range (0,len(decoded_adv)):
#         out_str = np.argmax(cap_predicted[j])
#         #print("out_str: ",out_str)
#         out_ = int(out_str)
#         #out_ = str(out_)
#         out_cap = in_vocab.idx2word[out_str]
#         caption_str = caption_str + " " + out_cap
    
#     print("caption_str: ", caption_str)
    
#     return encoded_img, caption_str
    