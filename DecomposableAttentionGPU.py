
# coding: utf-8

# In[276]:

import json
import random
import numpy as np
from collections import Counter
import pickle as pickle
import scipy.stats
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from preprocessing import * 

# ## Load Data and Embeddings

vocab_size = 50000
emb_dim = 300
num_classes = 3
learning_rate = .05
glove_path = 'glove.6B.300d.txt'
train_text_path = 'snli_1.0/snli_1.0_train.jsonl'
dev_text_path = 'snli_1.0/snli_1.0_dev.jsonl'
test_text_path = 'snli_1.0/snli_1.0_test.jsonl'

train_hypothesis, train_premise, train_label, train_label_enc = load_data(train_text_path)
dev_hypothesis, dev_premise, dev_label, dev_label_enc = load_data(dev_text_path)
test_hypothesis, test_premise, test_label, test_label_enc = load_data(test_text_path)

embeddings, words, idx2words, ordered_words = load_embeddings(glove_path, vocab_size, emb_dim)


#modifies embeddings, words, idx2words in place to add tokens
words, embeddings = add_tokens_da(words, embeddings, emb_dim)
idx2words = {v:k for k,v in words.items()}

h_len = 32 
p_len = 32 


train_h_idx = tokenize_da(train_hypothesis, words, h_len)
train_p_idx = tokenize_da(train_premise, words, p_len)

dev_h_idx = tokenize_da(dev_hypothesis, words, h_len)
dev_p_idx = tokenize_da(dev_premise, words, p_len)

test_h_idx = tokenize_da(test_hypothesis, words, h_len)
test_p_idx = tokenize_da(test_premise, words, p_len)


class DecomposableAttention(nn.Module):
    '''
    Starting with premise (a), we see if the hypothesis (b) is an 
    entailment, a contradiction, or neutral.
    '''
    def __init__(self, glove_emb, batch_size, hidden_size, h_len, p_len, num_classes, dropout=0.2):
        super(DecomposableAttention, self).__init__()
        self.glove = glove_emb
        self.num_embeddings = glove_emb.shape[0]
        self.embedding_dim = glove_emb.shape[1]
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.h_len = h_len
        self.p_len = p_len
        self.num_classes = num_classes
        self.dropout = dropout
        
        self.embed = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
        '''
        MLP LAYERS
        '''
        self.mlp_f = self._mlp_layers(self.hidden_size, self.hidden_size)
        self.mlp_g = self._mlp_layers(2 * self.hidden_size, self.hidden_size)
        self.mlp_h = self._mlp_layers(2 * self.hidden_size, self.hidden_size)
        
        '''
        Linear Layers
        '''
        self.project_h = nn.Linear(self.embedding_dim,self.hidden_size)
        self.project_p = nn.Linear(self.embedding_dim,self.hidden_size)
        self.final_linear = nn.Linear(self.hidden_size,self.num_classes)
        self.init_weights()
    
    def _mlp_layers(self, input_dim, output_dim):
        mlp_layers = []
        mlp_layers.append(nn.Dropout(p=self.dropout))
        mlp_layers.append(nn.Linear(input_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(p=self.dropout))
        mlp_layers.append(nn.Linear(output_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())   
        #sequential runs all the layers in order
        return nn.Sequential(*mlp_layers)  
    
    def forward(self, hypothesis, premise, label):
        start_time = time.time()
        
        '''
        Get padding masks
        Need to be LongTensors to avoid overflow with byte tensors
        '''
        h_mask = (hypothesis!=0).long()
        p_mask = (premise!=0).long()
        
        '''
        Embedding layer (only projection layer is trained)
        max length = max length of of hypothesis/premise (respectively) in batch
        Input dim: batch size x max length
        Output dim: batch size x max length x hidden dimensions
        '''
        p_embedded = self.embed(Variable(premise))
        h_embedded = self.embed(Variable(hypothesis))
        #project from embedding dim to hidden dim
        p_projected = self.project_p(p_embedded.view(-1,self.embedding_dim))                                         .view(self.batch_size,-1,self.hidden_size)
        h_projected = self.project_h(h_embedded.view(-1,self.embedding_dim))                                         .view(self.batch_size,-1,self.hidden_size)
        
        '''
        First Feed Forward Network (F)
        max length = max length of of hypothesis/premise (respectively) in batch
        Input dim: batch size x max length x hidden dimensions
        Output dim: batch size x max length x hidden dimension
        '''
        
        '''
        NEW MULTILAYER PERCEPTRON
        '''
        F_a = self.mlp_f(p_projected.view(-1, self.hidden_size)).view(self.batch_size,-1,self.hidden_size)
        F_b = self.mlp_f(h_projected.view(-1, self.hidden_size)).view(self.batch_size,-1,self.hidden_size)
        
        
        #E dim: batch_size x max len of hypothesis x max len of premise
        #transpose function swaps second and third axis so that F_b is batch size x hidden dim x len premise
        E = torch.matmul(F_a,torch.transpose(F_b,1,2))  
        
        '''
        Attention! 
        Given E, we reweight using the softmax and store in W_beta, W_alpha
        W_beta dim: batch_size x len(premise) x hidden dim
        W_alpha dim: batch_size x len(hypothesis) x hidden dim
        
        OLD:
        W_beta = Variable(torch.Tensor(self.batch_size,self.p_len,self.hidden_size))
        W_alpha = Variable(torch.Tensor(self.batch_size,self.h_len,self.hidden_size))
        for i in range(self.batch_size):
            for j in range(F_b.size()[1]):
                W_beta[i,j] = torch.mm(F.softmax(E[i,j]).view(1,-1),h_projected[i]).data
            for k in range(F_a.size()[1]):
                W_alpha[i,j] = torch.mm(F.softmax(E[i,:,j]).view(1,-1),p_projected[i]).data
        
        '''
        #p_mask is batch_size x p_len
        mask_a = p_mask.unsqueeze(1) #unsqueeze makes it batch_size x 1 x p_len
        mask_a = mask_a.expand_as(E.transpose(1, 2)).float() #expands it to (batch_size*h_len)x p_len
        mask_a = Variable(mask_a.view(-1, self.p_len))
        #mask_a.requires_grad = False
        
        mask_b = h_mask.unsqueeze(1) #unsqueeze makes it batch_size x 1 x h_len
        mask_b = mask_b.expand_as(E).float()  #expands it to (batch_size*p_len)x h_len
        mask_b = Variable(mask_b.view(-1, self.h_len))
        #mask_b.requires_grad = False
        
        #alpha is softmax over premise
        #dim: batch_size x h_len x p_len
        softmax_alpha = F.softmax(E.transpose(1, 2).contiguous().                                  view(-1, E.transpose(1, 2).size()[-1]))*mask_a
        #the +1e-13 is from allennlp. something about limiting numerical errors
        softmax_alpha = softmax_alpha / (softmax_alpha.sum(dim=1, keepdim=True) + 1e-13)
        softmax_alpha = softmax_alpha.view(E.transpose(1, 2).contiguous().size())
        
        #beta is softmax over the hypothesis
        #dim: batch_size x p_len x h_len
        softmax_beta = F.softmax(E.view(-1, E.size()[-1]))*mask_b
        softmax_beta = softmax_beta / (softmax_beta.sum(dim=1, keepdim=True) + 1e-13)
        softmax_beta = softmax_beta.view(E.size())
        
        
        '''
        softmax_beta is batch_size x p_len x h_len
        h_projected is batch size x h_len x hidden dimensions
        so W_beta is batch_size x p_len x hidden dim
        
        
        softmax_alpha is batch_size x h_len x p_len
        p_projected is batch size x p_len x hidden dimensions
        so W_alpha is batch size x h_len x hidden dime
        
        '''
        W_beta = torch.bmm(softmax_beta,h_projected)
        W_alpha = torch.bmm(softmax_alpha,p_projected)
        
        
        '''
        Compare
        Open items:
        1) Check that we're concatenating along the right dimensions.  Based on AllenNLP and libowen, 
            concatenated input should be batch size x len(hypothesis/premise) x (2 * embedding dim)
        
        Output:
        v1 dim: batch_size x len(hypothesis) x compare_dim
        v2 dim: batch_size x len(premise) x compare_dim
        '''
        #dim: batch size x len(hypotheis/premise) x (2* hidden dim)
        cat_p_beta = torch.cat((p_projected,W_beta),2)
        cat_h_alpha = torch.cat((h_projected,W_alpha),2)
        
        '''
        MLP with masking
        '''
        v_a = self.mlp_g(cat_p_beta.view(-1, 2*self.hidden_size))
        v_a = v_a*Variable(p_mask.view(-1).unsqueeze(1).expand_as(v_a).float())
        v_a = v_a.view(self.batch_size,-1,self.hidden_size)
        
        v_b = self.mlp_g(cat_h_alpha.view(-1, 2*self.hidden_size))
        v_b = v_b*Variable(h_mask.view(-1).unsqueeze(1).expand_as(v_b).float())
        v_b = v_b.view(self.batch_size,-1,self.hidden_size)
        '''
        Aggregate
        Given:
        v_a = output of relu activation on the concatenation of a (premise) and beta
        v_b = output of relu activation on the concatenation of b (hypothesis) and alpha
        '''
        v1 = torch.sum(v_a, dim=1)
        v2 = torch.sum(v_b, dim=1)
        
        H = self.mlp_h(torch.cat((v1,v2),1))
        out = self.final_linear(H)
        
        return out
    
    
    def init_weights(self):
        self.embed.weight.data.copy_(torch.from_numpy(self.glove))
        #does not train embedded weights
        self.embed.weight.requires_grad = False


#Iterator for training our model
#This generator gives 1 batch at a time
def batch_iter(dataset_size, hypothesis, premise, label_enc, batch_size, hLen, pLen):  
    start        = -1 * batch_size
    order        = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start     += batch_size
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
            random.shuffle(order)

        hBatch = torch.LongTensor(batch_size, hLen)
        pBatch = torch.LongTensor(batch_size, pLen)
        lBatch = torch.LongTensor(batch_size, 1)

        idx_list = order[start:start + batch_size]
        i = 0
        for idx in idx_list:
            hBatch[i] = torch.from_numpy(hypothesis[idx])
            pBatch[i] = torch.from_numpy(premise[idx])
            lBatch[i] = label_enc[idx]
            i += 1
            
        hBatch = hBatch.long().cuda()
        pBatch = pBatch.long().cuda()
        lBatch = Variable(lBatch).cuda()

        yield [hBatch, pBatch, lBatch]


#Iterator for evaluating our model 
#This generator gives a list of batches that can be iterated through.
def evaluation_iter(dataset_size, hypothesis, premise, label_enc, batch_size, hLen, pLen):
    batches = []
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while start < dataset_size - batch_size:
        start += batch_size
        
        hBatch = torch.LongTensor(batch_size, hLen)
        pBatch = torch.LongTensor(batch_size, pLen)
        lBatch = torch.LongTensor(batch_size, 1)

        idx_list = order[start:start + batch_size]
        i = 0
        for idx in idx_list:
            hBatch[i] = torch.from_numpy(hypothesis[idx])
            pBatch[i] = torch.from_numpy(premise[idx])
            lBatch[i] = label_enc[idx]
            i += 1
            
        hBatch = hBatch.long().cuda()
        pBatch = pBatch.long().cuda()
        lBatch = Variable(lBatch).cuda()
        
        if len(hBatch) ==  batch_size:
            batches.append([hBatch, pBatch, lBatch])
        else:
            continue       
    return batches


# This function outputs the accuracy on the dataset, we will use it during training.
def evaluate(model, data_iter, criterion):
    model.eval()
    correct = 0
    total = 0
    for i in range(len(data_iter)):
        hypothesis, premise, label = data_iter[i]
        
        output = model(hypothesis, premise, label)      
        _, predicted = torch.max(output.data, 1)
        lab = label.data.view(-1)
        total += lab.size(0)
        correct += (predicted == lab).sum()
       
	if i < 5:		 
        	print("\n ouput label: ")
        	print(output)
        	print("\n predicted")
        	print(predicted)
        	print("\n label")
        	print(lab.view(-1)) 
        	print("\n Sum: ")
        	print((predicted == lab).sum())
      
    return correct / float(total)



PATH='saved_model'
def training_loop(dataset_size, batch_size, num_epochs, model, data_iter, dev_iter, optimizer, criterion):
    model.train()
    step = 0
    epoch = 0
    losses = []
    total_batches = int(dataset_size / batch_size)
    print("\n total_batches: ", total_batches)
    while epoch <= num_epochs:
        hypothesis, premise, label = next(data_iter) 
        optimizer.zero_grad()
        output = model(hypothesis, premise, label)
        loss = criterion(output, label.view(-1))
        losses.append(loss.data[0])
        loss.backward()
        optimizer.step()

        if step % total_batches == 0:
            epoch += 1
            if epoch % 5 == 0:
                print( "Epoch:", (epoch), "Avg Loss:", np.mean(losses)/(total_batches*epoch),                        "Evaluate Loss: ", evaluate(model, dev_iter, criterion))
            	torch.save(model.state_dict(), PATH) # Saves model after every epoch
        step += 1
            

num_classes = 3
dropout_rate = .2
batch_size = 4
hidden_size = 200
h_len = 32 
p_len = 32 
num_epochs  = 40
learning_rate = .05

da = DecomposableAttention(embeddings,batch_size,hidden_size,h_len,p_len,num_classes,dropout=dropout_rate)


#filters out embedding layer which is not tuned
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, da.parameters()), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    da =  da.cuda()
    criterion = criterion.cuda()
else:
    print('cude unavailable')

#Train the Model
train_dataset_size = len(train_hypothesis)
data_iter = batch_iter(train_dataset_size, train_h_idx, train_p_idx, train_label_enc, batch_size, h_len, p_len)

dev_dataset_size = len(dev_hypothesis)
dev_iter = evaluation_iter(dev_dataset_size, dev_h_idx, dev_p_idx, dev_label_enc, batch_size, h_len, p_len)

training_loop(train_dataset_size, batch_size, num_epochs, da, data_iter, dev_iter, optimizer, criterion)

#Test the Model
test_dataset_size = len(test_hypothesis)
test_iter = evaluation_iter(test_dataset_size, test_h_idx, test_p_idx, test_label_enc, batch_size, h_len, p_len)
test_accuracy = evaluate(da, test_iter, criterion)

print('Accuracy of our model on the test data: %f' % (100 * test_accuracy))


'''Load saved model'''
#da = DecomposableAttention(embeddings,batch_size,hidden_size,h_len,p_len,num_classes,dropout=dropout_rate)
#da.load_state_dict(torch.load(PATH))

