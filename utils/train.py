import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np



def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points,range(len(points)))
    plt.show()


def eval_train(encoder,decoder,data_loader,accuracy):
    encoder.eval(),decoder.eval()
    accuracy_=0
    n_eval=0
    with torch.no_grad():
        for idx,(input_,targets) in enumerate(data_loader):
            _,encoder_hidden=encoder(input_)
            preds,attn_weights=decoder(encoder_hidden)
            for pred,target in zip(preds,targets):
                accuracy_+=accuracy(pred,target)
                n_eval+=1
    encoder.train(),decoder.train()
    return accuracy_.item()/n_eval


def train_bahdanau_luong_epoch(encoder,decoder,dataloader,criterion,encoder_optimizer,decoder_optimizer):
    total_loss=0
    for data in tqdm(dataloader):
        x,y=data
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        _,encoder_hidden=encoder(x)
        decoder_outputs,attn_weights=decoder(encoder_hidden,target_tensor=y)
        loss=criterion(decoder_outputs.reshape(-1,decoder_outputs.size(-1)),y.view(-1))
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss+=loss.item()
    return total_loss / len(dataloader)

def train_bahdanau_luong(epochs,encoder,decoder,train_dataloader,test_dataloader,lr,print_every,plot_every):
    plot_losses=[]
    print_losses=[]
    plot_loss_total=0
    print_loss_total=0
    decoder_optimizer=torch.optim.Adam(params=decoder.parameters(),lr=lr)
    encoder_optimizer=torch.optim.Adam(params=encoder.parameters(),lr=lr)
    criterion=nn.NLLLoss()
    accuracy=Accuracy(task="multiclass",num_classes=decoder.fcout.out_features,top_k=2).to(device)
    # pbar=tqdm(total=epochs,position=0,desc='Training Progress')
    # loss_log=tqdm(position=1,total=0,bar_foramt='{desc}')
    for i in range(1,epochs+1):
        avg_loss=train_bahdanau_luong_epoch(encoder,decoder,train_dataloader,criterion,encoder_optimizer,decoder_optimizer)
        val_acc=eval_train(encoder,decoder,test_dataloader,accuracy)
        plot_loss_total+=avg_loss
        print_loss_total+=avg_loss
        if i%print_every==0:
            print_losses.append(print_loss_total/print_every)
            print(f"Epoch {i} / {epochs} :  Loss {print_loss_total/print_every}   |   Validation Accuracy {val_acc}")
            print_loss_total=0
        # loss_log.set_description_str(f"Epoch {i} / {epochs} :  Loss {print_loss_total/print_every}")
        # pbar.update(1)
        if i%plot_every==0:
            plot_losses.append(plot_loss_total/plot_every)
            plot_loss_total=0
    showPlot(plot_losses)
    