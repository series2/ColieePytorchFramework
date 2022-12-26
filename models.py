import torch
from torch import nn
from torch.utils import data
import neptune.new as neptune
# import numpy as np

from utils import create_logger,create_neptune_run_instance
from logging import Logger
from base_model import TrainingModelBase,test_DataSet_DataLoader
from transformers import BertConfig, BertForSequenceClassification,BertModel
"""
check TODO
check Attention
"""


class SimpleDataset(data.Dataset):
    # 入るデータx,yはすでにtensorになっているとする。
    # to(device)はforwardの責務。
    def __init__(self,x,y): 
        self.X=x
        self.Y=y
    def __getitem__(self,index):
        return {key:self.X[key][index] for key in self.X.keys()} ,self.Y[index]
    def __len__(self):
        return len(self.Y) # xはdictなので、keyの数を返してしまう。


class PairDataset(data.Dataset):
    # 入るデータx,yはすでにtensorになっているとする。
    # to(device)はforwardの責務。

    # data["p"],data["h"],data["label"]を与える。
    # __get__item__ のi番目の0こめは、tokenizer(hi [SEP] pi)であり、shapeは(length,d) ここで、hi(判断文)が先頭に来る。
    # つまり、tokenizeまでやってしまう。padsequenceは全体で固定(バッジごとには行わない。)
    # xはtokenizerにかけ、paddingをしていると仮定する。

    # 以下は仮の仕様であり、ここでは使用しない。
    # 構造は、[CLS] t1 [SEP] t2 [SEP] [PAD] * である。 もし長い場合、t2を優先して切断する。
    # max_length番目が [PAD] の場合、スライスする。 max_length番目が[PAD]以外の時、max_lengthまで切断し、max_length番目に[SEP]を挿入する。
    # bert.tokenizerの使用ではmax_lengthによるtrancationはt1,t2を共に切断し、t2を優先して削除はできない(t2_onlyはあるが、t1が長すぎる場合error)
    def __init__(self,x,y): 
        self.X=x
        self.Y=y
    def __getitem__(self,index):
        return {key:self.X[key][index] for key in self.X.keys()} ,self.Y[index]
        # return {key:torch.tensor(self.X[key][index]) for key in self.X.keys()} ,torch.tensor(y[index])
    def __len__(self):
        return len(self.Y) # xはdictなので、keyの数を返してしまう。
    # def get_train_dataloader(self,batch_size,shuffle=False):
    #     def batch_colate_fn(batch):
    #         # batch_x ... [{},{}]
    #         if len(batch)==0:
    #             raise Exception("error batch length is 0")
    #         x={key:[] for key in batch[0].keys()}
    #         for xi in batch
    #             for key in xi.keys():
    #                 x[key].append(xi[key])
    #         return x
    #     return data.DataLoader(self,batsh_size,shuffle=shuffle,collate_fn=batch_colate_fn)
class DoubleDataset(data.Dataset):
    # data["p"],data["h"],data["label"]を与える。
    # __get__item__ のi番目の0個目は、{"p":tokenizer(p),"h":tokenizer(h)}である。
    # pi,hi をここで作る。つまり、tokenizeまでやってしまう。padsequenceは全体で固定(バッジごとには行わない。)
    def __init__(self,x,y): 
        self.X=x
        self.Y=y
    def __getitem__(self,index):
        return {key:self.X[key][index] for key in self.X.keys()} ,self.Y[index]
        # return {key:torch.tensor(self.X[key][index]) for key in self.X.keys()} ,torch.tensor(y[index])
    def __len__(self):
        return len(self.Y) # xはdictなので、keyの数を返してしまう。

class PlainBert(TrainingModelBase):
    def __init__(self,pretrained_model_name_or_path,logger:Logger=None,jlbert_token=None):
        super().__init__(logger=logger)
        bert_config=BertConfig.from_pretrained(pretrained_model_name_or_path,
                                                    gradient=True,
                                                    num_labels=2,
                                                    use_auth_token=jlbert_token
                                                    )
        self.bert=BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path,config=bert_config,use_auth_token=jlbert_token)
        self.logger.warning("it may use csv like `[CLS] col[0] SEP col[1]`. is it okey?")
    def forward(self,input_ids,attention_mask,token_type_ids,**other):
        # labels keyを無視する
        out=self.bert(**{"input_ids":input_ids,"attention_mask":attention_mask,"token_type_ids":token_type_ids})
        # self.logger.debug(out.logits)
        return {"CE":out.logits}

class MSCL(TrainingModelBase):
    def __init__(self,pretrained_model_name_or_path,logger:Logger=None,jlbert_token=None,is_ph_same_instance=True,use_da_times=1,tau=0.08):
        super().__init__(logger=logger)
        self.bert_config=BertConfig.from_pretrained(pretrained_model_name_or_path,
                                                    gradient=True,
                                                    # num_labels=2,使わないので。
                                                    use_auth_token=jlbert_token
                                                    )
        self.bert_p=BertModel.from_pretrained(pretrained_model_name_or_path,config=self.bert_config,use_auth_token=jlbert_token)
        self.bert_h=BertModel.from_pretrained(pretrained_model_name_or_path,config=self.bert_config,use_auth_token=jlbert_token) if not is_ph_same_instance else self.bert_p

        num_heads=12 # BERT内部のAttentionと同じサイズ
        self.attention=nn.MultiheadAttention(embed_dim=self.bert_config.hidden_size,num_heads=num_heads,batch_first=True) # out (B,L,D) when batch_first=True
        logger.warning(f"It use cross attention from torch.nn.MultiheadAttention.this is not different from original MSCL. And this model use num_heads={num_heads}") # add_bias_kvなどはない。 
        
        self.ff = nn.Sequential(
            nn.Linear(4*self.bert_config.hidden_size, self.bert_config.hidden_size),
            nn.ReLU())
        self.classifier = nn.Linear(4*self.bert_config.hidden_size, 2)
        self.use_da_times=use_da_times
        self.tau=tau
        assert tau!=0

        self.ce=nn.CrossEntropyLoss()
        self.cos_sim = nn.CosineSimilarity(dim=-1)

        self.logger.warning("Loss CE loss is applyed to Augmentated Data, So Consider Appropreate Epoch or change CE Loss for apply to only one data. ")
        self.logger.warning("SCL_Pair_Loss use dot_similarity (instead of cosine similarity)")

        # print("a" if not is_ph_same_instance else "b" )

    def forward(self,id_p,att_p,typ_p,id_h,att_h,typ_h,**other):
        def one_forward(id_p,att_p,typ_p,id_h,att_h,typ_h,**other):
            # (b,l) -> (b,l,d)
            b=id_p.shape[0]
            lp=id_p.shape[1]
            lh=id_h.shape[1]
            assert lp==lh , "For Sent Loss, token length of premise and hypothesis must be equal."
            d=self.bert_config.hidden_size

            encoded_p=self.bert_p(id_p,attention_mask=att_p,token_type_ids=typ_p,return_dict=True)["last_hidden_state"]
            encoded_h=self.bert_h(id_h,attention_mask=att_h,token_type_ids=typ_h,return_dict=True)["last_hidden_state"]
            assert encoded_p.shape==(b,lp,d) and encoded_h.shape==(b,lh,d)

            
            # (b,l,d) -> (b,l,d)
            attended_p=self.attention(encoded_p,encoded_h,encoded_h)[0] # attention return (attn_output,attn_output_weights) # query,key,value
            attended_h=self.attention(encoded_h,encoded_p,encoded_p)[0]
            assert attended_p.shape==(b,lp,d) and attended_h.shape==(b,lh,d)

            # (b,l,d) -> (b,l,4*d)
            enhanced_p = torch.cat([encoded_p, attended_p,
                                    encoded_p - attended_p,
                                    encoded_p * attended_p], dim=-1)
            enhanced_h = torch.cat([encoded_h, attended_h,
                                    encoded_h - attended_h,
                                    encoded_h * attended_h],dim=-1)
            assert enhanced_p.shape==(b,lp,4*d) and enhanced_h.shape==(b,lh,4*d)


            # (b,l,4*d) -> (b,l,d)
            projected_p = self.ff(enhanced_p)
            projected_h = self.ff(enhanced_h)
            assert projected_p.shape==(b,lp,d) and  projected_h.shape==(b,lh,d)

            # (b,l,d) -> (b,l,d)
            sp_hat=nn.functional.normalize(projected_p,dim=1)
            sh_hat=nn.functional.normalize(projected_h,dim=1)
            assert sp_hat.shape==(b,lp,d) and sp_hat.shape==(b,lh,d)

            # (b,l,d) -> (b,d)
            sp_mean=torch.mean(sp_hat,dim=1)
            sp_max,_index =torch.max(sp_hat,dim=1)
            sh_mean=torch.mean(sh_hat,dim=1)
            sh_max,_index= torch.max(sh_hat,dim=1)
            assert sp_mean.shape==(b,d) and sp_max.shape==(b,d) and sh_mean.shape==(b,d) and sh_max.shape==(b,d)

            # (b,d) -> (b,4*d)
            Z = torch.cat([sp_mean,sp_max,sh_mean,sh_max],dim=-1)
            assert Z.shape==(b,4*d)

            # (b,4*d) -> (b,2)
            logits=self.classifier(Z)
            assert logits.shape==(b,2)
            sent=torch.cat([encoded_p,encoded_h],dim=1) # batch方向は伸ばすので。 (b,2*l,d)

            return {"CE":logits,"Pair":Z,"Sent":sent}
        res={"CE":[],"Pair":[],"Sent":[]}
        times=self.use_da_times if self.training else 1
        for i in range(times):
            out=one_forward(id_p,att_p,typ_p,id_h,att_h,typ_h,**other)
            for key in out:
                res[key].append(out[key])
        for key in res:
            res[key]=torch.cat(res[key],dim=0)
        return res

    def CE_Loss(self,pred,gold):
        # self.logger.warning("this loss is applyed to Augmentated Data, So Consider Appropreate Epoch or change this Loss for apply to only one data. ")
        b=len(gold)
        # times=self.use_da_times if self.training else 1
        if self.training:
            gold=gold.repeat(self.use_da_times)
        assert gold.shape[0]==len(pred)
        loss=self.ce(pred,gold)
        assert not torch.isnan(loss).any(),f"{pred} ,\n{gold}"
        return loss

    def SCL_Pair_Loss(self,pred,gold):
        # pred ... Z (b,768*4=3072)
        """
        バッチ内部i番目についてgoldラベルの同じ全てのj(!=i)について、pred_iとpred_jでsimをとる。
        """
        eps=1e-10 # logのinf回避のための微小定数
        b=len(gold)
        if self.training:
            gold=gold.repeat(self.use_da_times)
        # pred shape ...(b*use_da_times,4*d) 
        # 類似度行列の次元は (b*use_da_times)**2なので大したことはない。
        # 類似度は論文では exp(dot(Zi,Zj)/tau) と cos_sim(Zi,Zj)/tau がある。Pair_SCEに合わせ、ここではdot内積を採用。
        matrix=torch.matmul(pred,pred.T)/self.tau 
        if self.training:
            assert matrix.shape==(b*self.use_da_times,b*self.use_da_times),f"wrong shape : {matrix.shape},{b}"
        else:
            assert matrix.shape==(b,b),f"wrong shape : {matrix.shape},{b}"
        # 注意 自分自身の内積も計算する。もしそれを省きたいなら、対角成分の計算グラフを切れば良い。また、類似度を負の無限大にすれば、expを撮った時0になる。
        # softmax_matrix=torch.nn.functional.softmax(matrix-torch.max(matrix,dim=1,keepdim=True)[0],dim=1)# 引き算はoverflow,underflow 対策。不要かもしれん。
        softmax_matrix=torch.nn.functional.softmax(matrix,dim=1)
        
        # self.logger.warning("SCL_Pair_Loss use dot_similarity (instead of cosine similarity)")
        # batch_iに対して、gold[i]と同じであるgold[j]のみ足し算する。
        # 例えば、i番目が(2,5,6,3)に対して 1,3番目を使いたいなら、dot((2,5,6,3),(True,False,True,False))/sum(True,False,True,False)=(2+6)/2 となる。

        tf=gold.unsqueeze(dim=0)==gold.unsqueeze(dim=1)  # shape (b(*use_da_times),b(*use_da_times)) 
        p=torch.sum(tf,dim=1,keepdim=True) # shape (b,1) # batch iと同じラベルがTrueになっている。自分自身を省いていないので1以上になるはず。
        assert torch.sum(p>0) ==len(p) ,f"{len(p)} , {torch.sum(p>0)},{p>0},{p}"

        
        
        li=-torch.log(torch.sum(softmax_matrix*tf,dim=1,keepdim=True)/p+eps) # shape (b,1)
        loss=torch.sum(li)
        assert not torch.isnan(loss).any()
        return loss
    
    def SCL_Sent_Loss(self,pred,gold):
        """
        leepは同じデータ拡張元を表す幅とする。
        バッチ内部i番目についてそれがTrueの時、Hiに対してPi及びP(i+leep)を正例とし、負例は存在しない。
        バッチ内部i番目について、それがFalseの時、Hiに対してPi及び(Pi+leep)、さらにはそれ以外の任意のデータを負例とし、正例は存在しない。
        なお、i番目がT/Fかはバッジによって変化するので、一般にロスは安定しなさそうである。
        一つ安定化手法で思いついたのは、 Trueのものについての cosine similarityは[-1-1]であるものがn子でありそれをTrueの数で割る。この値は大きくしたい。
        同様にしてFalseもそうする。この値は引き算したい。
        それぞれ正規化した後で max( True-False +1,0 ) とする。 1としたのは、True-Falseの最大値が2、最小値が-2であり、+側の半分を取った。
        これ良さそう。なのでそれで実装する。
        """
        margin=1.0
        b=len(gold)
        l=pred.shape[1]//2
        Sp,Sh=pred[:,:l,:],pred[:,l:,:]
        b_da=b if not self.training else b*self.use_da_times
        assert Sp.shape[0]==b_da,f"SP:{Sp.shape},{b},{self.use_da_times},{b_da},{self.training}"
            
        if self.training:
            gold=gold.repeat(self.use_da_times)
        # Sp,Sh shape ... (b,lp,d), (b,lh,d)
        self.logger.debug(f"{Sp.shape},{Sh.shape}")
        self.logger.debug(f"{b_da},{Sp.reshape((b_da,-1)).shape},{torch.reshape(Sp,(b_da,-1)).shape}")
        cossim_matrix=self.cos_sim(Sp.reshape((b_da,-1)).unsqueeze(1),Sh.reshape((b_da,-1)).unsqueeze(0))

        assert cossim_matrix.shape==(b_da,b_da),f"wrong shape : {cossim_matrix.shape},{b},{self.use_da_times},{b_da},{self.training}"

        leap_filter=torch.eye(b).to(gold.device) # todo
        if self.training:
            leap_filter=leap_filter.repeat(self.use_da_times,self.use_da_times)
        
        gold_filter=torch.unsqueeze((gold==1),dim=1) 
        # self.logger.debug(f"{gold_filter.shape},{leap_filter.shape}")
        poss_filter=leap_filter*gold_filter # shape (b,b)
        pos_n=torch.sum(poss_filter)
        poss=torch.sum(cossim_matrix*poss_filter)  # i業目について、gold[i]がTであるならば、j%leep = i なる 全てのj について足す。
        # これを得るには、以下のような周期的なフィルターと掛け合わせれば良い。
        """ b=3 でuse_da_times=2の場合
        same_filter gold_filter
        100100      111111
        010010      000000
        001001      000000
        100100      111111
        010010      000000
        001001      000000
        """
        
        neg_filter=torch.unsqueeze((gold==0),dim=1).repeat([1,b_da]) # shape(b_da,b_da)
        neg_n=torch.sum(neg_filter)
        negs=torch.sum(cossim_matrix*neg_filter) # i業目について、gold[i]がFであるならば、全てのjについて足す。

        self.logger.debug(f"cosssim_matrix.shap:{cossim_matrix.shape} , poss_filter.shape:{poss_filter.shape} , neg_filter.shape:{neg_filter.shape}")

        poss_rate=poss/pos_n if pos_n>0 else torch.zeros_like(pos_n)
        negs_rate=negs/neg_n if neg_n>0 else torch.zeros_like(neg_n)
        temp=(poss_rate - negs_rate) - margin # poss_rate-negs_rateを(marginより)大きくしたい。 つまり、tempは0を目指して大きくしたい。逆に言うと、-tempは0を目指して小さくしたい。# poss_rate=0.5,neg_rate=-0.5
        loss= max( -temp , torch.zeros_like(temp) ) # torch.zeros_like ... device ,dtypeも揃える。想定は size []
        # lossは 0を下らない。また、2+margin 以上にはならない。
        self.logger.debug(f"Sent_loss:{loss} , temp:{temp} , poss:{poss_rate}({poss}/{pos_n} , neg:{negs_rate}({negs}/{neg_n})")
        assert not torch.isnan(loss).any()
        return loss

class DACL(TrainingModelBase):
    def __init__(self,pretrained_model_name_or_path,logger:Logger=None,jlbert_token=None,use_da_times=1,tau=0.08):
        super().__init__(logger=logger)
        self.bert_config=BertConfig.from_pretrained(pretrained_model_name_or_path,
                                                    gradient=True,
                                                    num_labels=2,
                                                    use_auth_token=jlbert_token
                                                    )
        self.bert=BertModel.from_pretrained(pretrained_model_name_or_path,config=bert_config,use_auth_token=jlbert_token)

        self.ce=nn.CrossEntropyLoss()
        self.cos_sim = nn.CosineSimilarity(dim=1)
    
    def forward(self,id_p,att_p,typ_p,id_h,att_h,typ_h,labels,**other):
        pass

def plain_test():
    # DEBUG

    # Log SetUp Section
    logger=create_logger("model_debug",log_level=10) # 10 .. DEBUGも出力される。
    logger.debug("簡易的お試し用")

    # Device Section
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    logger.debug(f"device:{device}\n")

    model_name_or_path="cl-tohoku/bert-base-japanese-whole-word-masking"

    # Data Section
    logger.debug("Data Section")
    from transformers import BertJapaneseTokenizer, BertForMaskedLM
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
    texts=[("最初の前提文がここにあります。","最初の判断文はこちらです。"),("これは二つ目の前提文に相当します。","これは2つ目の仮定文です。"),("一応三つ目を置いとく","以下略")]
    x=tokenizer(texts,padding="max_length",max_length=40,truncation="only_second",return_tensors="pt")
    y=torch.tensor([1,0,1])
    logger.debug(x)

    # model Section
    logger.debug("model Section")
    model=PlainBert(model_name_or_path,logger=logger).to(device)
    logger.debug(model(**x))
    

    # DataLoader Section
    logger.debug("Data Loader Section")
    batch_size=2
    train_data=PairDataset(x,y)
    # train_dataloader=train_data.get_train_dataloader(batch_size=batch_size,shuffle=True)
    train_dataloader=data.DataLoader(train_data,batch_size,shuffle=True)
    valid_dataloader=train_dataloader
    test_x,test_y=x,y
    test_dataloader=data.DataLoader(train_data,batch_size,shuffle=False) # shuffleは必ずfalse
    test_DataSet_DataLoader(train_dataloader,logger)

    # exit(0)

    # ---------- Train Section start ----------
    logger.debug("Train Section start")
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-5) # , betas=(1-beta1,1-beta2),eps=eps,weight_decay=decay
    loss={"CE":nn.CrossEntropyLoss()} # CE use softmax in this function call(pred,gold)
    from sklearn.metrics import accuracy_score
    def accuracy(logits,gold):
        certainty_factor = logits[:, 1]
        # prediction = np.argmax(logits, axis=1)
        prediction=torch.argmax(logits,dim=1)
        return accuracy_score(gold.detach().cpu().numpy(), prediction.detach().cpu().numpy())
    model.compile(optimizer=optimizer, loss=loss, loss_weights={"CE":1.0}, metrics={"CE":accuracy})

    nep=create_neptune_run_instance(name="debug_pytorch",tags=["debug_pytorch"])
    output_dir="debug"
    model.fit(train_dataloader,valid_dataloader,epochs=5,output_dir=output_dir,nep=nep,save_monitor="CE",device=device) 
    model.test(test_x,test_y,test_dataloader,use_key="CE",output_dir=output_dir,device=device,nep=nep)
    nep.stop()
    logger.debug(f"Train Section end")
    # ---------- Train Section end ----------

if __name__=="__main__":
    plain_test()

