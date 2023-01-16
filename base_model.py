import torch
from torch import nn
from torch.utils import data
import tqdm
from collections import OrderedDict
import neptune.new as neptune
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from pprint import pformat
from logging import Logger

from utils import create_logger
"""
check TODO
check Attention
"""
# このプログラムでは loggingは、 raise Exception , tqdm , logger の3種あり、統一されていない。
class TrainingModelBase(nn.Module):
    # 子クラスにおいて、forwardメソッドを実装する。
    # loss_weightsがない時はloss全体で1になるような均等な割合にする。

    def __init__(self,logger:Logger=None):
        super().__init__()
        self.logger:Logger=logger if logger !=None else create_logger("TrainingModelBaseLogger")

    def forward(self,**kwarg): # **kwarg ... 辞書の形式で入力を受け取る kwarg が辞書型になる。
        raise Exception("In SubClass Implement")

    def compile(self,optimizer:torch.optim.Optimizer,loss:dict,metrics:dict,loss_weights:dict=None):
        if len(loss.keys())<=0:
            raise Exception("loss must be set at least one function")
        self.optimizer:torch.optim.Optimizer=optimizer
        self.loss_fns:dict=loss # {"loss_name",loss(gold,pred)}
        self.loss_weights:dict=loss_weights if loss_weights!=None else {key:1.0/len(loss.keys()) for key in loss.keys()}
        self.metrics_fns:dict=metrics
    def fit(self,train_dataloader,valid_dataloader,epochs,output_dir,nep:neptune.metadata_containers.Run=None,nep_base_namespace="",save_best_only=True,save_monitor="",device=torch.device('cpu'),reset=False):
        """
            neptuneには、_loss または _accがkeyの語尾に接続される。
            save_best_only は、save_monitorを基準にして、最高の値が出た時に上書きする。
            save_monitorには、validation_lossについて、saveに使用するkeyを与える。それ以外の場合未実装。lossの場合、これまでのepochで最小の値であるモデルを保存する。
            epoch などは 0スタートであり、保存したモデルを途中から始めることを想定しない。そのためsave_monitorも初期値は-inf,infなどである。
            dataloaderは辞書のx{feature1:torch.tensor([data1,data2,...],feature2:torch.tensor([data1,data2,..]))}と、
            配列のy torch.tensor([label1,label2,....])で構成される。
            return は monitor_loss(epochごとのvalid_lossらの中での最小値)
        """
        self.logger.warning("Do Implement for TQDM all loss")
        # self.nep=nep
        # self.nep_base_namespace=nep_base_namespace
        if save_best_only==False:
            raise Exception("Not Implementation")
        if save_monitor not in self.loss_fns.keys():
            raise Exception("save_monitor must be validation_loss key.") 
        if nep!=None and nep_base_namespace=="":
            nep_base_namespace=output_dir
        
        self.steps=0

        def train(dataloader,device,now_epoch=None):
            # global __steps
            size = len(dataloader.dataset)
            count=0
            log_loss_dict,log_acc_dict={key:0 for key in self.loss_fns.keys()},{key:0 for key in self.metrics_fns.keys()}
            self.train()
            disp="[Train]" if now_epoch==None else f"[Train:{now_epoch:>2}]"
            with tqdm.tqdm(dataloader,total=len(dataloader),desc=disp) as dataloader_with_progress_bar:
                for X,y in  dataloader_with_progress_bar:
                    self.steps+=1
                    count+=len(y)
                    loss_dict = {key:0 for key in self.loss_fns.keys()}
                    X={key:value.to(device) for key,value in X.items()}
                    y=y.to(device)

                    # Compute prediction error
                    # self.logger.warning(X)
                    out = self(**X) # return dict_output
                    for key,loss_fn in self.loss_fns.items():
                        # self.logger.debug(out[key])
                        # self.logger.debug(y)
                        assert not torch.isnan(out[key]).any(), f"loss {key}"
                        temp=loss_fn(out[key],y) # Attention pred,gold
                        assert not torch.isnan(temp).any(), f"loss {key}"
                        loss_dict[key]+=temp
                    loss=0
                    for key in loss_dict.keys():
                        loss+=self.loss_weights[key] * loss_dict[key]
                    # Backpropagation
                    self.optimizer.zero_grad()
                    assert not torch.isnan(loss).any() , loss_dict
                    loss.backward()
                    self.optimizer.step()

                    with torch.no_grad():
                        for key in self.loss_fns.keys():
                            log_loss_dict[key]+=loss_dict[key].item()*len(y)
                        for key,metrics_fn in self.metrics_fns.items():
                            log_acc_dict[key]+=metrics_fn(out[key],y).item()*len(y) # Attention (pred,gold)
                        od=OrderedDict()
                        assert count>0
                        for key,li in log_loss_dict.items():
                            od[f"train_{key}_loss"]=li/count
                        for key,mi in log_acc_dict.items():
                            od[f"train_{key}_acc"]=mi/count
                        # od["train_loss"]=
                        dataloader_with_progress_bar.set_postfix(od)
            # with torch.no_grad(): itemでnumpyなどに変換しているので、不要
            assert size>0
            # assert count==size, f"{count},{size}" # 内部で拡張がある場合、size = len(dataloader.dataset)より大きくなる。
            for key in self.loss_fns.keys():
                log_loss_dict[key]/=count
            for key,metrics_fn in self.metrics_fns.items():
                log_acc_dict[key]/=count
            del out
            del loss
            return log_loss_dict,log_acc_dict
                # if batch % 100 == 0:
                #     loss, current = loss.item(), batch * len(y)
                #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        def valid(dataloader,device,now_epoch=None):
            size = len(dataloader.dataset)
            count=0
            self.eval()
            log_loss_dict,log_acc_dict={key:0 for key in self.loss_fns.keys()},{key:0 for key in self.metrics_fns.keys()}
            with torch.no_grad():
                disp="[Valid]" if now_epoch==None else f"[Valid:{now_epoch:>2}]"
                with tqdm.tqdm(dataloader,total=len(dataloader),desc=disp) as dataloader_with_progress_bar:
                    for X, y in dataloader_with_progress_bar:
                        count+=len(y)
                        X={key:value.to(device) for key,value in X.items()}
                        y=y.to(device)

                        out = self(**X)
                        for key,loss_fn in self.loss_fns.items():
                            assert not torch.isnan(out[key]).any(), f"loss {key}"
                            temp=loss_fn(out[key],y) # Attention pred,gold
                            assert not torch.isnan(temp).any(), f"loss {key}"
                            log_loss_dict[key]+=temp.item()*len(y) # Attention pred,gold
                        for key,metrics_fn in self.metrics_fns.items():
                            log_acc_dict[key]+=metrics_fn(out[key],y)*len(y) # Attention pred,gold
                        od=OrderedDict()
                        assert count>0
                        for key,li in log_loss_dict.items():
                            od[f"val_{key}_loss"]=li/count
                        for key,mi in log_acc_dict.items():
                            od[f"val_{key}_acc"]=mi/count
                        dataloader_with_progress_bar.set_postfix(od)
            assert size>0
            for key in self.loss_fns.keys():
                log_loss_dict[key]/=count
            for key,metrics_fn in self.metrics_fns.items():
                log_acc_dict[key]/=count
            return log_loss_dict,log_acc_dict
        
        monitor_loss=sys.float_info.max # 注意 epoch などは 0スタートであり、保存したモデルを途中から始めることを想定しない。
        for t in tqdm.tqdm(range(epochs),desc="[Epoch]"):
            log_train_loss,log_train_acc=train(train_dataloader,device,t+1)
            log_valid_loss,log_valid_acc=valid(valid_dataloader,device,t+1)

            if log_valid_loss[save_monitor]<monitor_loss:
                monitor_loss=log_valid_loss[save_monitor]
                model_path=os.path.join(output_dir,"model.pth") #  'BERT_Task4.h5'
                torch.save(self.state_dict(), model_path)
                self.logger.info(f"Saved PyTorch Model State to {model_path}")
            if nep!=None:
                train_loss=0
                for key in log_train_loss.keys():
                    train_loss+=self.loss_weights[key] * log_train_loss[key]
                if t==0 and reset:
                    nep[os.path.join(nep_base_namespace,"train/epoch","loss")].pop()
                nep[os.path.join(nep_base_namespace,"train/epoch","loss")].log(train_loss)
                val_loss=0
                for key in log_valid_loss.keys():
                    val_loss+=self.loss_weights[key] * log_valid_loss[key]
                if t==0 and reset:
                    nep[os.path.join(nep_base_namespace,"validation/epoch","loss")].pop()
                nep[os.path.join(nep_base_namespace,"validation/epoch","loss")].log(val_loss)

                for key,value in log_train_loss.items():
                    if t==0 and reset:
                        nep[os.path.join(nep_base_namespace,"train/epoch",f"{key}_loss")].pop()
                    nep[os.path.join(nep_base_namespace,"train/epoch",f"{key}_loss")].log(value)
                for key,value in log_train_acc.items():
                    if t==0 and reset:
                        nep[os.path.join(nep_base_namespace,"train/epoch",f"{key}_acc")].pop()
                    nep[os.path.join(nep_base_namespace,"train/epoch",f"{key}_acc")].log(value)
                for key,value in log_valid_loss.items():
                    if t==0 and reset:
                        nep[os.path.join(nep_base_namespace,"validation/epoch",f"{key}_loss")].pop()
                    nep[os.path.join(nep_base_namespace,"validation/epoch",f"{key}_loss")].log(value)
                for key,value in log_valid_acc.items():
                    if t==0 and reset:
                        nep[os.path.join(nep_base_namespace,"validation/epoch",f"{key}_acc")].pop()
                    nep[os.path.join(nep_base_namespace,"validation/epoch",f"{key}_acc")].log(value)
        if nep!=None:
            nep[os.path.join(nep_base_namespace,"fit_params/epochs")]=epochs
            nep[os.path.join(nep_base_namespace,"fit_params/steps")]=self.steps
            nep[os.path.join(nep_base_namespace,"model/summary")]=str(self)
            # for key,value in self.optimizer.state_dict().items():
            #     nep[os.path.join(nep_base_namespace,"model/optimizer_config",key)]=value
            nep[os.path.join(nep_base_namespace,"model/optimizer_config/param_groups")]=self.optimizer.state_dict()["param_groups"]
            for key,value in self.loss_weights.items():
                nep[os.path.join(nep_base_namespace,"fit_params/loss_weights",key)]=value
        self.logger.info("End model.fit()")
        # self.steps=0

        return monitor_loss
    
    def load_weights(self,path):
        # self.load_state_dict(torch.load(path,map_location=torch.device("cpu")))
        self.load_state_dict(torch.load(path))


    def predict(self,test_dataloader,device): # return dict of torch tensor
        size = len(test_dataloader.dataset)
        self.eval()
        result_dict=None
        with torch.no_grad():
            with tqdm.tqdm(test_dataloader,total=len(test_dataloader),desc="[Test]") as dataloader_with_progress_bar:
                for X, y in dataloader_with_progress_bar:
                    X={key:value.to(device) for key,value in X.items()}
                    y=y.to(device)
                    out = self(**X)
                    # self.logger.debug(out)
                    out={key:value for key,value in out.items()}
                    if result_dict==None:
                        result_dict={key:[] for key in out.keys()}
                    for key in out.keys():
                        result_dict[key].append(out[key])
        # self.logger.debug(result_dict)
        result_dict={key:torch.cat(value,dim=0) for key,value in result_dict.items()} 
        # self.logger.debug(result_dict)
        return result_dict
    
    def test(self,test_x,test_y,test_dataloader,use_key,output_dir,device,nep:neptune.metadata_containers.Run=None,nep_base_namespace=""):
        """
            use_keyで使用する際、2値分類であり、out[key]のshapeは(batch,2)であり、out[key][:,1]が test_yの1に対応すると仮定する。
            もし異なるにしたい時はoverrideする。
        """
        if nep!=None and nep_base_namespace=="":
            nep_base_namespace=output_dir

        out=self.predict(test_dataloader,device)
        logit=out[use_key]
        # print(len(test_x),len(test_y),logit.shape)
        certainty_factor = nn.functional.softmax(logit,dim=1)[:, 1].detach().cpu().numpy()
        # prediction = np.argmax(logit, axis=1)
        prediction=torch.argmax(logit,dim=1).detach().cpu().numpy()
        # TODO CFは現状確率ではないので修正する
        pd.Series(certainty_factor).to_csv(os.path.join(output_dir,'test_CF.tsv'), header=None, index=False)
        pd.Series(prediction).to_csv(os.path.join(output_dir,'test_prediction.tsv'), header=None, index=False)
        self.logger.debug(logit.detach().cpu().numpy().shape)
        # self.logger.debug(test_x)
        self.logger.debug(len(test_dataloader.dataset))
        self.logger.debug(test_y)
        self.logger.debug(prediction)
        report=classification_report(test_y, prediction, digits=4,output_dict=True) # Attention (gold,pred)
        self.logger.info(pformat(report))
        report_df=pd.DataFrame(report).T
        report_df.to_csv(os.path.join(output_dir,"test_report.csv"))
        if nep!=None:
            nep[os.path.join(nep_base_namespace,"test/report")]=report_df
            nep[os.path.join(nep_base_namespace,"test/acc")]=report['accuracy']
        return report['accuracy']


def test_DataSet_DataLoader(dataloader,logger,trim=3):
    count=0
    for data in dataloader:
        logger.debug(data)
        # print(data) # data ... [x_batch,y_batch]
        count+=1
        if count>=trim:
            break

