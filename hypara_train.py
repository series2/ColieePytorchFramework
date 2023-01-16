import pandas as pd
import os
import optuna
from utils import load_dataset,create_logger, create_neptune_run_instance
# train,testはいかなるモデルに対しても同一(fix)である。それゆえこの関数は1回しか使われない。
def create_data(files,save_dir,train=0.8,valid=0.1,test=0.1):
    os.mkdir(save_dir)
    assert train+valid+test==1
    dfs=[]
    for j,file_name in enumerate(files):
        file_name=os.path.join(file_name) 
        df=pd.read_table(file_name,dtype='object', na_filter=False,header=None)
        if len(df.columns)!=3 and len(df.columns)!=4:
            raise Exception("Concat Failed.")
        df=df.rename(columns={0:"label",1:"premise",2:"hypothesis"})
        dfs.append(df)
    df=pd.concat(dfs,axis=0)
    df = df[["label","premise","hypothesis"]] # extentionは、一旦なしで。
    df=df.sample(frac=1) # shuffle
    train_size=int(len(df)*train)
    valid_size=int(len(df)*valid)
    test_size=len(df)-train_size-valid_size
    df_train=df[:train_size]
    df_valid=df[train_size:-test_size]
    df_test=df[-test_size:]

    df_train.to_csv(os.path.join(save_dir,"train.tsv"), sep='\t', header=None, index=None)
    df_valid.to_csv(os.path.join(save_dir,"valid.tsv"), sep='\t', header=None, index=None)
    df_test.to_csv(os.path.join(save_dir,"test.tsv"), sep='\t', header=None, index=None)
    
    print("train_size",len(df_train)) #644
    print("valid_size",len(df_valid)) # 80
    print("test_size",len(df_test)) # 82


# train_file , aug_file , valid_file , train_file 及び　なんらかのハイパラを使用して、optunaによるハイパラserrchをする。
# 最も良いepochの選択には、valid_fileのメトリクスを使用する。
# それのメトリクスの値を、その時のハイパラにおけるスコアとする。
# すると、任意のスコアに対して最適なepcohでのvalid_metricsを得ることができる。
# それで最適なhyparaを選択する。
# test_fileは最後にモデルの指標を図る上で数回するなどして利用する。
# ここにあるスクリプトは全て
# Attention hyparaについて、seedは固定しない。
import train
class Objective:
    def __init__(self,train,valid,output_dir_prefix,arch,logger_name,batch_size=None,epochs=None,max_len=None,lr=None,debug=False,seed=None,aug_path=None,test_path=None,**args):
        self.logger=create_logger(logger_name)
        if debug:
            self.logger.debug(f"\n\n{'-'*10}debug mode{'-'*10}\n\n")
        self.logger.warning("Objective Class Only support arch MSCL. Do Impoment")
        if arch!="MSCL":
            raise Exception("Only Suppoer MSCL.") # TODO
        if debug:
            epochs=2
        if epochs==None:
            epochs=5 if arch!="MSCL" else 10
        if max_len==None:
            max_len=512 if arch=="plain" else 256 # else ... double,MSCL,DACL
            if arch=="MSCL":
                max_len=256 # max_lenはMSCLではメモリ使用量には関係ありそう。TODO
            self.logger.info(f"max_len is automatically setted to {max_len} in model {arch}")
        
        if batch_size==None:
            batch_size=12 if arch!="MSCL" else 12
            # batch_sizeはMSCLではメモリ使用量には関係なさそう。
            self.logger.info(f"batch_size is automatically setted to {batch_size} in model {arch}")

        if lr==None:
            lr=1e-5
            if arch!="plain":
                self.logger.warning(f"learning rate({lr}) may be not good. in model {arch}") # TODO
                # logger.warning(f"batch_size({batch_size}) may be not good. in model {arch}") # TODO
        self.train_file_path=train
        self.valid_file_path=valid
        self.output_dir_prefix=output_dir_prefix
        self.arch=arch
        self.debug=debug
        self.seed=seed
        self.aug_path=aug_path
        self.test_path=test_path
        self.args=args

        self.batch_size=batch_size
        self.epochs=epochs
        self.max_len=max_len
        self.lr=lr

        self.nep=create_neptune_run_instance(logger_name)
        self.nep_base_name=""
        

    def __call__(self,trial):
        # MSCLのhypara
        ce_loss_weight = 1 # 固定
        pair_loss_weight=trial.suggest_float("pair_loss_weight", 0.0, 1.0) # TODO range と 層
        sent_loss_weight=trial.suggest_float("sent_loss_weight", 0.0, 1.0) # TODO range と 層
        use_in_model_da_times=2

        output_dir=f"{self.output_dir_prefix}_pair-{pair_loss_weight}_sent-{sent_loss_weight}"
        os.mkdir(output_dir)
        args={}
        args.update(self.args)
        args.update({
                "ce_loss_weight":ce_loss_weight,
                "pair_loss_weight":pair_loss_weight,
                "sent_loss_weight":sent_loss_weight,
                "use_in_model_da_times":use_in_model_da_times,
            })
        pretrained_model_name_or_path="cl-tohoku/bert-base-japanese-whole-word-masking"
        metrics=train.train(train_file_path=self.train_file_path,output_dir=output_dir,arch=self.arch,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            test_path=self.test_path,aug_path=self.aug_path,valid_file=self.valid_file_path,
            batch_size=self.batch_size,epochs=self.epochs,max_len=self.max_len,lr=self.lr,
            nep=self.nep,nep_base_namespace=self.nep_base_name,
            jlbert_token=None,debug=self.debug,logger=self.logger,seed=self.seed,
            **args)
        
        return metrics
    
    def stop(self):
        self.nep.stop()

def do_optuna_hypara_search(study,storage,output_dir_prefix):

    study = optuna.create_study(study_name=study_name,
                                storage=storage,
                                direction="minimize",
                                load_if_exists=True) # 複数プロセスで動かす。なお、dbサーバーは同じ(ストレージファイルとして存在)なので、mysql serverを立てなくてもsqliteで良い。
    train_file_path="/work/source/data/coliee/random_split_tsv/train.tsv"
    valid_file_path="/work/source/data/coliee/random_split_tsv/valid.tsv"
    arch="MSCL"
    test_path=None #"work/source/data/coliee/random_split_tsv/test.tsv" # 今回は不要
    aug_path="/work/source/data/civil/tsv/coliee_2019.tsv"
    # aug_path=None
    arch="MSCL"
    debug=False
    # debug=True
    seed=None
    args={}
    logger_name="MSCL_hyparasearch"
    objective=Objective(train_file_path,valid_file_path,output_dir_prefix,arch,logger_name,debug=debug,aug_path=aug_path,test_path=test_path,**args)
    n_trials=100 if not debug else 3
    study.optimize(objective, n_trials=n_trials) # defaultは optuna tpe sampler
    objective.stop()
    # print(f"最良の誤差: {study.best_value}")
    # print(f"最良のハイパーパラメータ: {study.best_params}")

def show_optuna_results(study_name,storage,output_dir_prefix):
    from preprocessing import convert_examples_to_features,convert_examples_to_multi_bert_features
    from models import PlainBert,MSCL,SimpleDataset
    from transformers import BertJapaneseTokenizer
    import torch
    from torch.utils.data import DataLoader
    print("warning only support MSCL.")
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage,
        load_if_exists=True,
    )
    print(f"全体で最良の誤差: {study.best_value}")
    print(f"全体で最良のハイパーパラメータ: {study.best_params}")
    # https://app.neptune.ai/o/tokyotech-cl/org/COLIEE-ito/e/COL-602/all?path=work%2Fsource%2FColieePytorchFramework%2Fexperiment2_results%2FMSCL%2FMSCL_pair-0.030365261757214834_sent-0.016706093097935956%2F
    # return study
    if True:
        trials=list(filter(lambda x:x.value!=None,study.trials))
        trials=sorted(trials, key=lambda x:x.value)
        size=len(trials)
        trials=trials[:10]
        results=[]
        for i,trial in enumerate(trials):
            print(f"{i+1}番めに良いモデルのの誤差: {trial.value}")
            print(f"{i+1}番めに良いモデルのハイパーパラメータ: {trial.params}")
            output_dir=f"{output_dir_prefix}_pair-{trial.params['pair_loss_weight']}_sent-{trial.params['sent_loss_weight']}"
            pretrained_model_name_or_path="cl-tohoku/bert-base-japanese-whole-word-masking"
            model_path=f"{output_dir}/model.pth"
            # pretrained_model_name_or_path=model_path
            tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model_name_or_path) # TODO JLBERT
            # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            device=torch.device('cpu')
            # model=MSCL(model_path)  # 初期化されるのでダメ。
            model = MSCL(model_path).to(device)
            model.load_state_dict(torch.load(model_path))
            test_path="/work/source/data/coliee/random_split_tsv/test.tsv"
            x_test, y_test = load_dataset(test_path,debug=False)  # テストデータをロード
            max_len=256
            batch_size=12
            x_feature,y_feature=convert_examples_to_multi_bert_features(x_test,y_test,max_seq_length=max_len,tokenizer=tokenizer)
            data=SimpleDataset(x_feature,y_feature)
            test_dataloader=DataLoader(data,batch_size,shuffle=False)
            acc=model.test(x_test,y_test,test_dataloader,use_key="CE",output_dir=output_dir,device=device)
            print(acc)
            results.append([trial.value,trial.params,acc])
            print()
        
        print("全trial",size)
        for i,trial in enumerate(trials):
            print(f"{i+1}番めに良いモデルのの誤差: {results[i][0]}")
            print(f"{i+1}番めに良いモデルのハイパーパラメータ: {results[i][1]}")
            print(f"acc:{results[i][2]}")
            print()

if __name__ == "__main__":
    create_flag=False
    if create_flag:
        DATA_FILE_PATHS=[
            "/work/source/data/coliee/tsv/riteval_H18_jp.tsv",
            "/work/source/data/coliee/tsv/riteval_H19_jp.tsv",
            "/work/source/data/coliee/tsv/riteval_H20_jp.tsv",
            "/work/source/data/coliee/tsv/riteval_H21_jp.tsv",
            "/work/source/data/coliee/tsv/riteval_H22_jp.tsv",
            "/work/source/data/coliee/tsv/riteval_H23_jp.tsv",
            "/work/source/data/coliee/tsv/riteval_H24_jp.tsv",
            "/work/source/data/coliee/tsv/riteval_H25_jp.tsv",
            "/work/source/data/coliee/tsv/riteval_H26_jp.tsv",
            "/work/source/data/coliee/tsv/riteval_H27_jp.tsv",
            "/work/source/data/coliee/tsv/riteval_H28_jp.tsv",
            "/work/source/data/coliee/tsv/riteval_H29_jp.tsv",
            "/work/source/data/coliee/tsv/riteval_H30_jp.tsv",
            "/work/source/data/coliee/tsv/riteval_R01_jp.tsv"
            # "riteval_R02_jp.tsv", 
        ] #H18-R02まで15年分
        create_data(DATA_FILE_PATHS,"/work/source/data/coliee/random_split_tsv")
    
    study_name="MSCL_search"

    storage="sqlite:///example.db"
    output_dir_prefix="/work/source/ColieePytorchFramework/experiment2_results/MSCL/MSCL"

    # do_hypara_search=True
    do_hypara_search=False
    if do_hypara_search:
        do_optuna_hypara_search(study_name,storage,output_dir_prefix)

    # show_results=False
    show_results=True
    if show_results:
        show_optuna_results(study_name,storage,output_dir_prefix)

