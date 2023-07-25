import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


######################BAHDANAU ATTENTION######################

class EncoderBiRNN(nn.Module):
    def __init__(self,vocab_size,num_layers,input_size,hidden_size):
        super(EncoderBiRNN,self).__init__()
        self.embedding=nn.Embedding(vocab_size,input_size)
        self.BiRNN=nn.GRU(input_size,hidden_size, num_layers=num_layers,bidirectional=True,batch_first=True)
        
    def forward(self,x,hidden=None):
        N,max_L=x.shape
        hiddens=[]
        outputs=[]
        for i in range(max_L):
            embedding=self.embedding(x[:,i].unsqueeze(1))
            output,hidden=self.BiRNN(embedding,hidden)
            outputs.append(output)
            hiddens.append(hidden.unsqueeze(0))
        outputs=torch.cat(outputs,dim=1)            # outputs [max_L,N,1,enc_hidden_size]
        hiddens=torch.cat(hiddens,dim=0)            # hiddens [max_L,enc_nlayers*directions,N,enc_hidden_size]

        return outputs,hiddens
    
    #encoding_dict={directions=enc_directions,
        # enc_hidden_size=enc_hidden_size,
        # dec_hidden_size=enc_hidden_size,
        # enc_num_layers=enc_num_layers,
        # dec_num_layers=dec_num_layers}

        


class BahdanauAttention(nn.Module):
    def __init__(self,encoding_dict):
        super(BahdanauAttention,self).__init__()
        self.encoding_dict=encoding_dict
        self.Wa=nn.Linear(self.encoding_dict['dec_hidden_size']*self.encoding_dict['dec_num_layers'],self.encoding_dict['dec_hidden_size'])
        self.Ua=nn.Linear(self.encoding_dict["enc_hidden_size"],self.encoding_dict['dec_hidden_size'])
        self.Va=nn.Linear(self.encoding_dict["dec_hidden_size"],1)

    def forward(self,query,keys): 
        #query [dec_num_layers,N,dec_hidden]; keys  [max_L,(enc_nlayers*directions),N,enc_hidden_size]  
        
        #  [dec_num_layers,N,dec_hidden]-> [N,dec_num_layers,dec_hidden]-> [N,1,dec_num_layers*dec_hidden]
        query=query.transpose(0,1)
        query=query.reshape(query.size(0),1,query.size(1)*query.size(2))    
        #  [max_L,(enc_nlayers*directions),N,enc_hidden_size] -> [max_L*(enc_nlayers*directions),N,enc_hidden_size] -> [N,max_L*(enc_nlayers*directions),enc_hidden_size]
        keys=keys.reshape(-1,keys.size(2),keys.size(3)).transpose(0,1)
        
        #  query: [N, 1, dec_num_layers*dec_hidden] @ [dec_hidden*dec_num_layers, dec_hidden] -> [N, 1, dec_hidden]
        #  keys:  [N, max_L*(enc_nlayers*directions), enc_hidden_size] @ [enc_hidden_size, dec_hidden] -> [N, max_L*(enc_nlayers*directions), dec_hidden]
        
        #  score(addition): [N, max_L*(enc_nlayers*directions), dec_hidden] + [N, 1, dec_hidden](broadcasted) -> [N, max_L*(enc_nlayers*directions), dec_hidden]
        #  score(reduce_sum ): [N, max_L*(enc_nlayers*directions), dec_hidden] @ [dec_hidden, 1] -> [N, max_L*(enc_nlayers*directions), 1]                
        scores=self.Va(torch.tanh(self.Wa(query)+self.Ua(keys)))
        #  score: [N, max_L*(enc_nlayers*directions), 1] -> [N, max_L*(enc_nlayers*directions)] -> [N, 1, max_L*(enc_nlayers*directions)]  
        scores=scores.squeeze().unsqueeze(1)
        
        #softmax(weights)
        weights=F.softmax(scores,dim=-1)
        #  [N,1,max_L*(enc_nlayers*directions)] @ [N,max_L*(enc_nlayers*directions),enc_hidden_size] -> [N,1,enc_hidden_size]
        context=torch.bmm(weights,keys) 

        return context,weights

class BahdanauDecoder(nn.Module):
    def __init__(self,vocab_size,hidden_size,num_layers,encoding_dict):
        super(BahdanauDecoder,self).__init__()
        self.encoding_dict=encoding_dict
        self.encoding_dict['dec_hidden_size']=hidden_size                
        self.encoding_dict['dec_num_layers']=num_layers
        self.embedding=nn.Embedding(vocab_size,hidden_size)                            # Embedding Layer: [ vocab_size, enc_hidden_dims ]
        self.attention=BahdanauAttention(self.encoding_dict)                           # Inputs: [t-1_dec_hidden_state,all_enc_hidden_states] | Outputs: [context_vector,attention_weights]
        self.gru=nn.GRU(
            input_size=self.encoding_dict['enc_hidden_size']+self.encoding_dict['dec_hidden_size'],       #Inputs: [context+input_concat,decoder_hidden_state]
            hidden_size=self.encoding_dict['dec_hidden_size'],                                       #Outputs: [lstm_final_layer_activations,dec_hidden_state]
            num_layers=self.encoding_dict['dec_num_layers'],
            batch_first=True
            )
        self.fcout=nn.Linear(self.encoding_dict['dec_hidden_size'],vocab_size)

    def forward(self,encoder_hiddens,target_tensor=None):
        MAX_LENGTH,N=encoder_hiddens.size(0),encoder_hiddens.size(2)
        decoder_input=torch.empty((N,1),dtype=torch.long).fill_(SOS_token).to(device)                   #create empty input
        decoder_outputs=[]  #output token cache
        attention_weights=[]  #atttention weights cache
        decoder_hidden=torch.zeros((self.encoding_dict['dec_num_layers'],N,self.encoding_dict['dec_hidden_size'])).to(device)
        for i in range(MAX_LENGTH):
            decoder_output,decoder_hidden,attn_weight=self.forward_step(decoder_input,encoder_hiddens,decoder_hidden)
            if target_tensor!=None:
                decoder_input=target_tensor[:,i].unsqueeze(1)               #Teacher Forcing with groundtruth label inputs
            else:
                _,topi=decoder_output.topk(1)                               #Predicted output to input
                decoder_input=topi.squeeze(-1).detach()
            
            
            
            decoder_outputs.append(decoder_output)
            attention_weights.append(attn_weight)
        decoder_outputs=F.log_softmax(torch.cat(decoder_outputs,dim=1),dim=-1)
        attention_weights=torch.cat(attention_weights,dim=1)
        return decoder_outputs,attention_weights
    
    def forward_step(self,input,encoder_states,decoder_hidden):
        embedded=self.embedding(input)          #[batch_size,1] -> [batch_size,1,decoder_hidden]
        context,weights=self.attention(query=decoder_hidden,keys=encoder_states) #[[dec_num_layers,N,dec_hidden];[max_L,(enc_nlayers*directions),N,enc_hidden_size]] ->[N,1,enc_hidden_size];[N,1,max_L]    
        nn_inp=torch.cat([context,embedded],dim=-1)                         # [N,1,enc_hidden_size+dec_hidden]
        decoder_output,decoder_hidden=self.gru(nn_inp,decoder_hidden)       # [[N,1,enc_hidden_size+dec_hidden], [dec_num_layers,N,dec_hidden]] -> [N,1,dec_hidden_state],[dec_num_layers,N,dec_hidden]                                              
        decoder_output=self.fcout(decoder_output)                           # [N,1,dec_hidden_state] -> [N,1,output_vocab_size]
        return decoder_output,decoder_hidden,weights                






######################LUONG ATTENTION######################
class EncoderLSTM(nn.Module):
    def __init__(self,vocab_size,input_size,hidden_size,num_layers):
        super(EncoderLSTM,self).__init__()
        self.embedding=nn.Embedding(vocab_size,input_size)
        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
    def forward(self,x,state_cell=None):
        hiddens=[]
        N,max_L=x.shape
        for i in range(max_L):
            embedded=self.embedding(x[:,i].unsqueeze(1))
            if not state_cell:
                _,state_cell=self.lstm(embedded)
            else:
                _,state_cell=self.lstm(embedded,state_cell)
            hidden=state_cell[0]
            hiddens.append(hidden.unsqueeze(2))
        hiddens=torch.cat(hiddens,dim=2)                        # we use only the last hidden layers of both encoder and decoder hidden_states
        return None,hiddens[-1]                  # encoder_hidden_state (last layer) [N, MAX_L, encoder_hidden_dims]

# encoding_dict={
#     'enc_hidden_size':enc_hidden_size,
#     'enc_num_layers': enc_num_layers,
#     'dec_hidden_size':dec_hidden_size,
#     "dec_num_layers":dec_num_layers
#     
#     
# }

class dotAttention(nn.Module):
    def __init__(self,_):                   #added _ for consistency with rest of the multiplicative techniques
        super(dotAttention,self).__init__()
    def forward(self,dec_hidden_state,enc_hidden_states):                               
        dot=torch.bmm(enc_hidden_states,dec_hidden_state.transpose(-1,-2)).transpose(-1,-2)
        # dot=torch.einsum("N1H,NLH->N1L",[dec_hidden_state,enc_hidden_states])
        attn_w=F.softmax(dot,dim=-1)        # NIL @
        context=torch.bmm(attn_w,enc_hidden_states)
        return context,attn_w                       #context: N,L,eH
        
class generalDotAttention(nn.Module):
    def __init__(self,encoding_dict):
        super(generalDotAttention,self).__init__()
        self.W_a=nn.Linear(encoding_dict['enc_hidden_size'],encoding_dict['dec_hidden_size'])
    def forward(self,dec_hidden_state,enc_hidden_states):
        enc_hidden_alligned=self.W_a(enc_hidden_states)
        dec_hidden_state=dec_hidden_state.transpose(-1,-2)                    ## N,1,H -> N,H,1
        dot=torch.bmm(enc_hidden_alligned,dec_hidden_state).transpose(-1,-2)  ## N,L,H @ N,H,1 -> N,L,1 -> N,1,L
        attn_w=F.softmax(dot,dim=-1)        # N,1,L
        context=torch.bmm(attn_w,enc_hidden_states)  ## N,1,L @ N,L,eH
        return context,attn_w                               #context: N,1,eH
        
class concatAttention(nn.Module):
    def __init__(self,encoding_dict):
        super(concatAttention,self).__init__()
        self.W_a=nn.Linear(encoding_dict['enc_hidden_size']+encoding_dict['dec_hidden_size'],encoding_dict['dec_hidden_size'])
        self.v_a=nn.Linear(encoding_dict['dec_hidden_size'],1)
    def forward(self,dec_hidden_state,enc_hidden_states):
        print(dec_hidden_state.shape,enc_hidden_states.shape)
        concat_=torch.concat((dec_hidden_state.expand(dec_hidden_state.size(0),enc_hidden_states.size(1),dec_hidden_state.size(2)),enc_hidden_states),dim=-1)      #N,1,dH -> N,L,dH c N,L,eH -> N,L,dH+eH  (dH==H)
        merged=torch.tanh(self.W_a(concat_))                #N,L,dH+eH -> N,L,H        
        attn_w=F.softmax(self.v_a(merged).transpose(-1,-2),dim=-1)  #N,L,H -> softmax(N,1,L)
        context=torch.bmm(attn_w,enc_hidden_states)     ## N,1,L @ N,L,eH
        return context,attn_w                     #context: N,1,eH ; attn_w:N,1,L


class LuongAttention(nn.Module):
    def __init__(self,encoding_dict:dict,max_L_d:Optional[tuple]=None,variant:str='general',type_:Optional[str]='global'):
        super(LuongAttention,self).__init__()
        self.attn_={'concat':concatAttention,'general':generalDotAttention,'dot':dotAttention}
        self.attention=self.attn_[variant](encoding_dict)
        self.type_=type_
        if max_L_d:
            max_L,d=max_L_d
            if d>max_L:
                raise Exception("Sorry, but local span cannot be greater than the max sequence length")
            self.seq_len=max_L
            self.d=d
            self.W_p=nn.Linear(encoding_dict['dec_hidden_size'],1)
            self.V_p=nn.Linear(encoding_dict['max_L'],1)


    def local_p(self,decoder_hidden_state):
        dec_sum=torch.tanh(self.W_p(decoder_hidden_state)).squeeze().unsqueeze(0)
        L_ratio=torch.sigmoid(self.V_p(dec_sum))
        return self.seq_len*L_ratio.squeeze()

    def forward(self,dec_hidden_state,enc_hidden_states):
        if self.type_=='local':
            p=self.local_p(dec_hidden_state)
            enc_hidden_states=enc_hidden_states[:,p-self.d:p+self.d+1,:]                    #[N,max_L,hidden_dims] -> [N,p-d:p+d,hidden_dims]
        context,attn_weights=self.attention(dec_hidden_state,enc_hidden_states)
        return context,attn_weights
    

class LuongDecoder(nn.Module):
    def __init__(self,vocab_size,hidden_size,num_layers,attn_variant,attn_type,encoding_dict):
        super(LuongDecoder,self).__init__()
        encoding_dict['dec_hidden_size']=hidden_size
        encoding_dict['dec_num_layers']=num_layers
        self.embedding=nn.Embedding(vocab_size,hidden_size)
        self.lstm=nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        self.attention=LuongAttention(encoding_dict,variant=attn_variant,type_=attn_type)
        self.W_c=nn.Linear(encoding_dict["enc_hidden_size"]+encoding_dict["dec_hidden_size"],hidden_size)
        self.fcout=nn.Linear(hidden_size,vocab_size)
        
    def forward(self,enc_hidden_states,target_tensor=None):
        N,max_L=enc_hidden_states.shape[:2]
        decoder_input=torch.empty((N,1),dtype=torch.long,device=device).fill_(SOS_token)
        dec_hidden_cell=None
        dec_outputs,attn_weights=[],[]
        for i in range(max_L):
            decoder_output,dec_hidden_cell,attn_w=self.forward_step(decoder_input,enc_hidden_states,dec_hidden_cell)
            if target_tensor==None:
                _,topi=decoder_output.topk(1)                               # N,1,vocab_size -> N,1,1                      
                decoder_input=topi.squeeze(-1).detach()                     # N,1,1 -> N,1 
            else:
                decoder_input=target_tensor[:,i].unsqueeze(1)               # N -> N,1
            dec_outputs.append(decoder_output)
            attn_weights.append(attn_w)
        dec_outputs=F.log_softmax(torch.cat(dec_outputs,dim=1),dim=-1)
        attn_weights=torch.cat(attn_weights,dim=1)
        return dec_outputs,attn_weights            


    def forward_step(self,input,enc_hidden_states,dec_hidden_cell=None):
        embedded=self.embedding(input)
        if not dec_hidden_cell:
            _,dec_hidden_cell=self.lstm(embedded)
        else:
            _,dec_hidden_cell=self.lstm(embedded,dec_hidden_cell)
        dec_hidden_state=dec_hidden_cell[0][-1].unsqueeze(1)      #(hidden,cell) -> hidden: Nlayers,N,dH -> N,dH ->  N,1,dH 
        context,attn_w=self.attention(dec_hidden_state,enc_hidden_states)
        context_cat=torch.cat([context,embedded],dim=-1)            # N,1,dH ; N,1,eH -> N,1,dH+eH
        decoder_output=torch.tanh(self.W_c(context_cat))            # N,1,dH+eH -> N,1,dH
        # decoder_output,decoder_hidden_cell=self.lstm(decoder_output,dec_hidden_cell)
        decoder_output=self.fcout(decoder_output)                   # N,1,dH -> N,1,vocab_size
        return decoder_output,dec_hidden_cell,attn_w

        

            

        