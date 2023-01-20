# 入力を同じ学習をしたBERTに入れて含意タスクを解く
# パラメタは非対称性を考慮して重みを別々にして学習する
# 最終層の2つの出力埋め込みから1次元への変換を施す。
from transformers import set_seed
from transformers import BertJapaneseTokenizer
from sklearn.model_selection import train_test_split
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from models import PlainBert,MSCL,DACL,SimpleDataset,TextDataset
from utils import load_dataset,create_logger,create_neptune_run_instance
import neptune.new as neptune
from get_controller_args import get_controller_args,str2dict_args_from_sh
from preprocessing import convert_examples_to_features,convert_examples_to_multi_bert_features

print("script train_script/train.py is loaded!")



def controller(train_file_path:str,arch:str,
        name,base_dir:str="./",
        loop_time:int=1,use_model_iter:bool=False,model_remove:bool=False,
        pretrained_model_name_or_path:str="cl-tohoku/bert-base-japanese-whole-word-masking",
        test_path:str=None,aug_path:str=None,valid_path:str=None,
        batch_size:int=None,epochs:int=None,max_len:int=None,lr:float=1e-5,
        without_neptune=False,neptune_instance=None,neptune_init_tags=[],reuse_nep=False, #reuse_nep ... idがあれば使う。
        jlbert_token:str=None,debug:bool=False,logger=None,reset=False,**args): # argsにはモデル固有のパラメタを入れる。
        # resetはエラーが起きるなどして再度学習をやり直したいときに使用する。
        ## argsの中身　use_in_model_da_times, (CELossWeight,PairLossWeight,SentLossWeight)
    """ 
    ある一年分の学習を1回または複数回回す。(loop_time)
    ただし、複数回す場合、モデルについては 
        "pretrained_model_name_or_path"_{i} iは1からloop_timeまで
    以上の形式に合うような形で毎回変更することができる。これは、同じ条件のもと事前学習モデルを複数作った時に使用することを想定する。
    neptune_instanceについては、例えば交差検証的な感じで実験するなど、controller関数を複数呼ぶ場合、neptune インスタンスを引き継ぎつぐこともできる。
    """
    if logger==None:
        logger=create_logger("controller")
    if debug:
        epcohs=5
        # epochs=2
    
    if batch_size==None:
        batch_size=12 if arch!="MSCL" else 12
        # batch_sizeはMSCLではメモリ使用量には関係なさそう。
        logger.info(f" is automatically setted to {batch_size} in model {arch}")

    if epochs==None:
        epochs=5 if arch!="MSCL" else 10
        if arch=="MSCL":
            logger.warning("epoch may be better if set to more than 5.") # TODO MSCLはもう少しepoch増やすべき?
    
    if max_len==None:
        max_len=512 if arch=="plain" or arch=="DACL" else 256 # else ... double,MSCL,DACL
        if arch=="MSCL":
            max_len=256 # max_lenはMSCLではメモリ使用量には関係ありそう。TODO
        logger.info(f"max_len is automatically setted to {max_len} in model {arch}")

    if arch!="plain":
        logger.warning(f"learning rate({lr}) may be not good. in model {arch}") # TODO
        # logger.warning(f"batch_size({batch_size}) may be not good. in model {arch}") # TODO

    if loop_time<=0:
        raise Exception("回数の入力が不正です")
    if debug:
        logger.debug(f"\n\n{'-'*10}debug mode{'-'*10}\n\n")
    # os.mkdir(os.path.join(base_dir,name)) # 存在していた場合error
    os.makedirs(os.path.join(base_dir,name),exist_ok=True)

    nep_create_flag=False
    if without_neptune:
        neptune_instance=None
    else:
        if neptune_instance==None:
            nep_create_flag=True
            nep_file=os.path.join(base_dir,"nep_id.txt")
            if reuse_nep:
                if os.path.isfile(nep_file):
                    with open(nep_file,"r") as f:
                        nep_id=f.read()
                        nep_id=nep_id.rstrip()
                    logger.info(f"use nep_id ... {nep_id}")
                    logger.warning("reuse nep instance. but have not used the 'create_neptune_run_instance'. so you cannot use wrapped neptune.")
                    from dotenv import load_dotenv
                    load_dotenv(override=True)
                    neptune_instance = neptune.init_run(with_id=nep_id,
                        project=os.getenv('NEPTUNE_PROJECT'),
                        api_token=os.getenv('NEPTUNE_API_TOKEN'))
                else:
                    tags=[]
                    tags.extend(neptune_init_tags)
                    if debug:
                        tags.append("Debug")
                    neptune_instance=create_neptune_run_instance(name=base_dir,tags=tags)
                    with open(nep_file,"w") as f:
                        f.write(neptune_instance["sys/id"].fetch())
            else:
                tags=[name]
                # if "tags" in args.keys():
                    # tags.extend(args["tags"])
                tags.extend(neptune_init_tags)
                if debug:
                    tags.append("Debug")
                neptune_instance=create_neptune_run_instance(name=name,tags=tags)
        params={
                "train_file_path":train_file_path,
                "arch":arch,
                "name":name,
                "base_dir":base_dir,
                "loop_time":loop_time,
                "use_model_iter":use_model_iter,
                "model_remove":model_remove,
                "pretrained_model_name_or_path":pretrained_model_name_or_path,
                "test_path":test_path,
                "valid_path":valid_path,
                "aug_path":aug_path,
                "augmentation":(aug_path!=None),
                "batch_size":batch_size,
                "epohcs":epochs,
                "max_len":max_len,
                "lr":lr,
                "jlbert_token":(jlbert_token!=None),
                "debug":"debug",
                "Optimizer":"Adam",
                "reset":reset,
            }
        params.update(args) # 辞書型 argを追加する
        neptune_instance[os.path.join(name,"parameters")] = params

        
    if loop_time==1:
        output_dir=os.path.join(base_dir,name)
        if use_model_iter:
            pretrained_model_name_or_path=f"{pretrained_model_name_or_path}-1"
        nep_base_namespace=name
        train(train_file_path=train_file_path,output_dir=output_dir,arch=arch,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            test_path=test_path,aug_path=aug_path,valid_path=valid_path,
            batch_size=batch_size,epochs=epochs,max_len=max_len,lr=lr,
            nep=neptune_instance,nep_base_namespace=nep_base_namespace,
            jlbert_token=jlbert_token,debug=debug,logger=logger,reset=reset,**args)
        if model_remove:
                os.remove(os.path.join(output_dir,"model.pth"))
    else:
        for i in range(1,loop_time+1):
            output_dir=os.path.join(base_dir,name,str(i))
            if use_model_iter:
                pretrained_model_name_or_path=f"{pretrained_model_name_or_path}-{i}"
            os.mkdir(output_dir)
            nep_base_namespace=os.path.join(name,str(i))
            train(train_file_path=train_file_path,output_dir=output_dir,arch=arch,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                test_path=test_path,aug_path=aug_path,valid_path=valid_path,
                batch_size=batch_size,epochs=epochs,max_len=max_len,lr=lr,
                nep=neptune_instance,nep_base_namespace=nep_base_namespace,
                jlbert_token=jlbert_token,debug=debug,logger=logger,reset=reset,**args)
            if model_remove:
                os.remove(os.path.join(output_dir,"model.pth"))
    if nep_create_flag:
        neptune_instance.stop()


# 1回の学習。
def train(train_file_path:str,output_dir:str,arch:str,
        pretrained_model_name_or_path:str="cl-tohoku/bert-base-japanese-whole-word-masking",
        test_path=None,aug_path=None,valid_path=None,# validは与えなければtrainから分割される
        batch_size:int=12,epochs:int=5,max_len:int=256,lr:float=1e-5,
        nep=None,nep_base_namespace=None,
        jlbert_token=None,debug=False,logger=None,seed=None,reset=False,**args):
    if logger==None:
        logger=create_logger("train_one_time")
    support_arch=["plain","double","MSCL","DACL"]
    if arch not in support_arch:
        raise Exception(f"model {arch} is out of support.")

    if seed==None:
        random.seed()
        seed=random.randint(0, 500)
    logger.info(f"seed is set to {seed}")
    set_seed(seed)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    # TODO
    # =======================================================================================================================================

    logger.debug(f"device:{device}")


    if nep!=None:
        if nep_base_namespace==None:
            raise Exception("nep_base_name have to be set.")
        nep[os.path.join(nep_base_namespace,"run_seed")]=seed

    x_train, y_train = load_dataset(train_file_path,debug)  # 訓練データをロード

    if valid_path==None:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=random.randint(0, 2**31 - 1))
    else:
        x_val,y_val=load_dataset(valid_path,debug)

    if aug_path:
        x_art, y_art = load_dataset(aug_path,debug)  # 拡張データをロード
        x_train.extend(x_art)
        y_train.extend(y_art)
    if test_path:
        x_test, y_test = load_dataset(test_path,debug)  # テストデータをロード

    if jlbert_token!=None:
        from pyknp import Juman
        class JumanppTokenizer:
            def __init__(self):
                self.jumanpp = Juman()

            def tokenize_jumanpp(self, cont_ja_str: str):
                morphemes = [mrph.midasi for mrph in self.jumanpp.analysis(cont_ja_str).mrph_list()]
                res_line = " ".join(morphemes) 
                return res_line
        jlbert_tokenizer = JumanppTokenizer()# 単語を分けるtokenizer
    # モデルをビルド
    tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model_name_or_path) # TODO JLBERT
    # 必ずLoss "CE" の名前でモデルを定義しておくこと。
    if arch=="plain":
        x_feature,y_feature=convert_examples_to_features(x_train,y_train,max_seq_length=max_len,tokenizer=tokenizer)
        data=SimpleDataset(x_feature,y_feature)
        train_dataloader=DataLoader(data,batch_size,shuffle=True)

        x_feature,y_feature=convert_examples_to_features(x_val,y_val,max_seq_length=max_len,tokenizer=tokenizer)
        data=SimpleDataset(x_feature,y_feature)
        valid_dataloader=DataLoader(data,batch_size,shuffle=True)

        if test_path:
            x_feature,y_feature=convert_examples_to_features(x_test,y_test,max_seq_length=max_len,tokenizer=tokenizer)
            data=SimpleDataset(x_feature,y_feature)
            test_dataloader=DataLoader(data,batch_size,shuffle=False)

        model=PlainBert(pretrained_model_name_or_path,logger=logger,jlbert_token=jlbert_token).to(device)
        optimizer=torch.optim.Adam(model.parameters(), lr=lr) # , betas=(1-beta1,1-beta2),eps=eps,weight_decay=decay
        def accuracy(logits:torch.Tensor,gold:torch.Tensor):
            prediction=torch.argmax(logits,dim=1)
            return accuracy_score(gold.detach().cpu().numpy(), prediction.detach().cpu().numpy())
        model.compile(optimizer=optimizer, loss={"CE":nn.CrossEntropyLoss()}, metrics={'CE':accuracy})
    elif arch=="double":
        raise Exception(f"not implemented arch {arch}") #TODO
         # x_train, x_val, y_train, y_val x_test, y_test 
    elif arch=="MSCL":
        # raise Exception(f"not implemented arch {arch} check when use_in_model_da_times>1, acc") #TODO
        special_params=["use_in_model_da_times","ce_loss_weight","pair_loss_weight","sent_loss_weight"]
        # optionとして、ce_use_embs,sent_only_use_clsも存在。
        for s_p in special_params:
            if s_p not in args.keys():
                raise Exception(f"you need args '{s_p}' for MSCL")
         # x_train, x_val, y_train, y_val x_test, y_test 
        x_feature,y_feature=convert_examples_to_multi_bert_features(x_train,y_train,max_seq_length=max_len,tokenizer=tokenizer)
        data=SimpleDataset(x_feature,y_feature)
        train_dataloader=DataLoader(data,batch_size,shuffle=True)

        x_feature,y_feature=convert_examples_to_multi_bert_features(x_val,y_val,max_seq_length=max_len,tokenizer=tokenizer)
        data=SimpleDataset(x_feature,y_feature)
        valid_dataloader=DataLoader(data,batch_size,shuffle=True)

        if test_path:
            x_feature,y_feature=convert_examples_to_multi_bert_features(x_test,y_test,max_seq_length=max_len,tokenizer=tokenizer)
            data=SimpleDataset(x_feature,y_feature)
            test_dataloader=DataLoader(data,batch_size,shuffle=False)
        use_da_times=args["use_in_model_da_times"] # 必須
        ce_use_emb=args.get("ce_use_embs",False)
        sent_only_use_cls=args.get("sent_only_use_cls",False)
        model=MSCL(pretrained_model_name_or_path,logger=logger,jlbert_token=jlbert_token,is_ph_same_instance=True,use_da_times=use_da_times,tau=0.08,ce_use_emb=ce_use_emb,sent_only_use_cls=sent_only_use_cls).to(device)
        optimizer=torch.optim.Adam(model.parameters(), lr=lr) # , betas=(1-beta1,1-beta2),eps=eps,weight_decay=decay
        def accuracy(logits:torch.Tensor,gold:torch.Tensor):
            b=len(gold) # logits (b*? , 2)
            prediction=torch.argmax(logits[:b,:],dim=1)
            return accuracy_score(gold.detach().cpu().numpy(), prediction.detach().cpu().numpy())
        model.compile(
            optimizer=optimizer,
            loss={
                "CE":model.CE_Loss,
                "Pair":model.SCL_Pair_Loss,
                "Sent":model.SCL_Sent_Loss,
                },
            loss_weights={
                "CE":args["ce_loss_weight"],
                "Pair":args["pair_loss_weight"],
                "Sent":args["sent_loss_weight"],
                },
                # TODO accuracy は 増やした方が良い??? TODO accuracyの関数を渡す。
        metrics={"CE":accuracy}) # 拡張データではやらない。(DAはDropoutなのでevalでは使われなさそう)つまり、一つ分。
    elif arch=="DACL":
        # raise Exception(f"not implemented arch {arch}")
        special_params=["use_in_model_da_times","pseudo_neg","ce_loss_weight","cl_loss_weight"]
        for s_p in special_params:
            if s_p not in args.keys():
                raise Exception(f"you need args '{s_p}' for MSCL")
        converter = lambda x,y:convert_examples_to_features(x,y,max_seq_length=max_len,tokenizer=tokenizer)
        use_da_times=args["use_in_model_da_times"] # 必須
        pseudo_neg=args["pseudo_neg"]
        model =DACL(pretrained_model_name_or_path,converter=converter,logger=logger,jlbert_token=jlbert_token,use_da_times=use_da_times,pseudo_neg=pseudo_neg,tau=0.08).to(device) # TODO
        data=TextDataset(x_train,y_train)
        train_dataloader=DataLoader(data,batch_size,collate_fn=model.dataloader_collate_fn,shuffle=True)
        data=TextDataset(x_val,y_val)
        valid_dataloader=DataLoader(data,batch_size,collate_fn=model.dataloader_collate_fn,shuffle=True)

        if test_path:
            data=TextDataset(x_test,y_test)
            test_dataloader=DataLoader(data,batch_size,collate_fn=model.dataloader_collate_fn,shuffle=False)
        optimizer=torch.optim.Adam(model.parameters(), lr=lr) # , betas=(1-beta1,1-beta2),eps=eps,weight_decay=decay
        model.compile(
            optimizer=optimizer,
            loss={
                "CE":model.CE_Loss,
                "CL":model.CL_Pair_Loss,
                },
            loss_weights={
                "CE":args["ce_loss_weight"],
                "CL":args["cl_loss_weight"],
                },
            metrics={"CE":model.accuracy}
        )
    else:
        raise Exception(f"not implemented arch {arch}")


    monitor_loss=model.fit(train_dataloader,valid_dataloader,epochs=epochs,output_dir=output_dir,nep=nep,
                    save_monitor="CE",device=device,nep_base_namespace=nep_base_namespace,reset=reset)

    model.load_weights(os.path.join(output_dir,"model.pth")) #  'BERT_Task4.h5'

    if test_path:
        model.test(x_test,y_test,test_dataloader,use_key="CE",output_dir=output_dir,device=device,nep=nep,nep_base_namespace=nep_base_namespace)

    # if nep!=None: # ここで開いたなら良いが、そうでないので勝手に閉じてはいけない。
        # nep.stop()
    
    del model  #明示的に解放(GPU　メモリー どんどん貯まる) plain_tohoku-bert_NA/DABaseは問題なかったが、MSCLがダメ。
    torch.cuda.empty_cache()

    return monitor_loss


"""
    cd train_script
    mkdir ../results/simcse_wiki_double_bert_without_augmentation
    python3 train.py -i ../augmentation/train.tsv -t ../augmentation/test.tsv -b  ../results -n ./simcse_wiki_double_bert_without_augmentation  -m /work/source/Logic-DrivenContextExtention/SimSCE/SimCSE/result/ja-wiki-unsup-simcse-bert-base-japanese-whole-word-masking

    for debug
    python3 train.py -i ../augmentation/train.tsv ../debugs -n ./simcse_wiki_double_bert_without_augmentation --debug --without_neptune
"""

def train_debug():
    max_len=12
    logger=create_logger("train_debug",log_level=10)
    
    logger.debug("train debug start")
    # Dataset Create Test
    train_file_path="/work/source/Logic-DrivenContextExtention/data/coliee/tsv/riteval_H18_jp.tsv"
    aug_path="/work/source/Logic-DrivenContextExtention/data/coliee/tsv/riteval_H18_jp.tsv"
    test_path="/work/source/Logic-DrivenContextExtention/data/coliee/tsv/riteval_H18_jp.tsv"

    model_name_or_path="cl-tohoku/bert-base-japanese-whole-word-masking"
    from transformers import BertJapaneseTokenizer
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
    x_train, y_train = load_dataset(train_file_path)
    logger.debug(x_train)
    logger.debug(y_train)
    x_feature,y_feature=convert_examples_to_features(x_train,y_train,max_seq_length=max_len,tokenizer=tokenizer)
    logger.debug(x_feature)
    logger.debug(y_feature)

     # DataLoader Section
    logger.debug("Data Loader Section")
    batch_size=12
    train_data=SimpleDataset(x_feature,y_feature)
    train_dataloader=DataLoader(train_data,batch_size=batch_size,shuffle=False)
    from base_model import test_DataSet_DataLoader
    test_DataSet_DataLoader(train_dataloader,logger)


    # Train Section
    logger.debug("Train Section")
    nep=create_neptune_run_instance("train_debug",tags=["debug"])
    output_dir="debug_train"
    os.makedirs(output_dir,exist_ok=True)
    train(train_file_path=train_file_path,output_dir=output_dir,arch="plain",
        pretrained_model_name_or_path=model_name_or_path,
        test_path=test_path,aug_path=aug_path,
        batch_size=batch_size,epochs=2,max_len=max_len,
        nep=nep,nep_base_namespace="train_debugggg",
        jlbert_token=None,debug=False,logger=logger)
    nep.stop()
    
    logger.debug("train debug end!")


def controller_debug():
    model_name_or_path="cl-tohoku/bert-base-japanese-whole-word-masking"
    logger=create_logger("train_debug")
    logger.debug("controler debug start!")
    train_file_path="/work/source/Logic-DrivenContextExtention/data/coliee/tsv/riteval_H18_jp.tsv"
    aug_path="/work/source/Logic-DrivenContextExtention/data/coliee/tsv/riteval_H19_jp.tsv"
    test_path="/work/source/Logic-DrivenContextExtention/data/coliee/tsv/riteval_H20_jp.tsv"
    logger.debug("Controller Section")
    controller(train_file_path=train_file_path,arch="plain",
        name="controller_test",base_dir="./",
        loop_time=2,use_model_iter=False,model_remove=False,
        pretrained_model_name_or_path=model_name_or_path,
        test_path=test_path,aug_path=aug_path,
        batch_size=12,epochs=2,max_len=64,
        without_neptune=False,neptuen_instance=None,
        jlbert_token=None,debug=False,logger=logger)
    logger.debug("controller debug end!")


def MSCL_train_test(seed):
    train(
    train_file_path="/work/source/PytorchFramework/results/MSCL_tohoku-bert_NA_debug/without_riteval_H18_jp/train.tsv",
    output_dir="/work/source/PytorchFramework/results/MSCL_tohoku-bert_NA_debug",
    arch="MSCL",
    seed=seed,
    use_in_model_da_times=1,
    ce_loss_weight=1.0,
    pair_loss_weight=0.0,
    sent_loss_weight=0.0
    )

def DACL_train_test():
    seed=0
    train(
        train_file_path="/work/source/data/coliee/tsv/riteval_H18_jp.tsv",
        output_dir="/work/source/ColieePytorchFramework/results/DACL_tohoku-bert_NA_debug",
        arch="DACL",
        seed=seed,
        use_in_model_da_times=2,# 2も試す
        pseudo_neg=2, # 1,2も試す
        ce_loss_weight=1.0,
        cl_loss_weight=0.0, # 0以上も試す
    )

def MSCL_Details_Test(bert_in,bert_model=None):
    if bert_model==None:
        from transformers import BertConfig,BertModel
        pretrained_model_name_or_path="cl-tohoku/bert-base-japanese-whole-word-masking"
        bert_config=BertConfig.from_pretrained(pretrained_model_name_or_path,
                                                        gradient=True,
                                                        # num_labels=2,使わないので。
                                                        # use_auth_token=jlbert_token
                                                        )
        bert_model=BertModel.from_pretrained(
                pretrained_model_name_or_path,
                config=bert_config,
                # use_auth_token=jlbert_token
                ).to("cuda:0")
    out=bert_model(**bert_in,return_dict=True)
    print("MSCL_Details_Test")
    print(out)

def do_debut_test():
    # seed=474 # Train 1 epoch  batch 39/58
    seed=24 # Train 1 epoch  batch 3/58
    # seed=309 # Train 1 epoch  batch 20/58
    # seed=3 # Train 1 epoch  batch 11/58
    # seed=413 # Train 1 epoch  batch 30/58
    # seed=69 # Train 1 epoch ok , 2 epoch ... 40/58
    # seed=None

    if seed==None:
        random.seed()
        seed=random.randint(0, 500)
    print(f"seed is set to {seed}")
    set_seed(seed)

    MSCL_train_test(seed)

if __name__ == '__main__':
    # train_debug()
    # controller_debug()

    args=get_controller_args()
    args=vars(args) # 辞書に変換
    controller(**args)

    # do_debut_test():

    # args=str2dict_args_from_sh()
    # controller(args)

    # DACL_train_test()

