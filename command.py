import os
import neptune.new as neptune
import sys
# import pprint
import pandas as pd
import argparse
from train import controller
from utils import create_logger,create_neptune_run_instance
from dotenv import load_dotenv

"""
check TODO
check Attention
"""

"""
このプログラムはtrain.pyのcontrollerを制御する。
学習データファイルのうち1ファイルをTest,それ以外をTrainにして回すことをn回ずつ全通りで実験できる。

このプログラムでは、学習ファイルは全て
{DATA_DIR_PATH}/{DATA_FILE_PATHS}
として存在する。このプログラムでは、
{DATA_FILE_PATHS}のうち、一つをTestデータとし、
それ以外の{DATA_FILE_PATHS}と、{AUGMENTATION_FILE_PATH}のデータを学習データとして使用する。

create_without_data()関数
{DATA_FILE_PATHS}のi番目をテストデータとして除外して、それ以外の{DATA_FILE_PATHS}をまとめて{save_path}に出力する。
{save_path}は、具体的には {base_dir}/without_{DATA_FILE_PATHS[i]}/train.csv である。

experiment_all_year()関数
base_dir/without_{DATA_FILE_PATHS[i]}/[j]/ でj(1,2,など)回目の実験の結果を出力する。

do_experiment_all_year_from_shell() 関数
if __name__=="__main__"から呼ばれる。
上記experiment_all_year関数を実行したい時にシェルスクリプトなどで引数を渡すための関数。

write_sh_script()関数
上記を実行するシェルスクリプトを出力する関数。
実験設定が複数ある時使用する。 command.shを作成する。
"""


load_dotenv(override=True) # このファイルのあるディレクトリ(あるいは上位)にある .envファイルを参照する。
"""
.envファイルフォーマット
NEPTUNE_PROJECT={YOUR_NEPTUNE_PROJECT}
NEPTUNE_API_TOKEN={YOUR_NEPTUNE_API_TOKEN}
DATA_DIR_PATH={YOUR_DATA_DIR_PATH}
AUGMENTATION_FILE_PATH={YOUR_AUGMENTATION_FILE_PATH}
もし存在しなくてもerroorにならないので、必須のDATA_DIR_PATH以外は書かなくても良い。
"""
DATA_DIR_PATH = os.getenv('DATA_DIR_PATH')
AUGMENTATION_FILE_PATH = os.getenv('AUGMENTATION_FILE_PATH')

DATA_FILE_PATHS=[
    "riteval_H18_jp.tsv",
    "riteval_H19_jp.tsv",
    "riteval_H20_jp.tsv",
    "riteval_H21_jp.tsv",
    "riteval_H22_jp.tsv",
    "riteval_H23_jp.tsv",
    "riteval_H24_jp.tsv",
    "riteval_H25_jp.tsv",
    "riteval_H26_jp.tsv",
    "riteval_H27_jp.tsv",
    "riteval_H28_jp.tsv",
    "riteval_H29_jp.tsv",
    "riteval_H30_jp.tsv",
    "riteval_R01_jp.tsv"
    # "riteval_R02_jp.tsv", 
] #H18-R02まで15年分



def create_without_data(files,i,save_path):
    dfs=[]
    for j,file_name in enumerate(files):
        if j==i:
            continue
        file_name=os.path.join(DATA_DIR_PATH,file_name) #このファイルからの相対パス
        df=pd.read_table(file_name,dtype='object', na_filter=False,header=None)
        if len(df.columns)==4: # extentionあり
            df=df.rename(columns={0:"label",1:"premise",2:"hypothesis",3:"extention"})
        elif len(df.columns)==3: # なし
            df=df.rename(columns={0:"label",1:"premise",2:"hypothesis"})
        dfs.append(df)
    df=pd.concat(dfs,axis=0)
    if len(df.columns)!=3 and len(df.columns)!=4:
        raise Exception("Concat Failed.")
    
    df = df[["label","premise","hypothesis"]] # extentionは、一旦なしで。
    df.to_csv(save_path, sep='\t', header=None, index=None) #  save_pathは実行スクリプトからの相対パス

def do_experiment_all_year_from_shell(write_only=False):
    # 今回指定すべき項目
    # architecture
    argparser = argparse.ArgumentParser()
    argparser.add_argument("execute_mode")
    argparser.add_argument('-b', '--base_dir', required=True)
    argparser.add_argument('--meta_name', required=True)
    argparser.add_argument('--architecture', required=True)
    argparser.add_argument('-m', '--model_path', required=True)
    argparser.add_argument('--augmentation_flag',action="store_true") # デフォルトはFalse
    argparser.add_argument('-l', '--loop_time',type=int,default=5)
    argparser.add_argument('--model_remove',action="store_true") # デフォルトFalse 
    argparser.add_argument('--neptune_tags',nargs="*") # 空白で複数
    argparser.add_argument('--debug',action="store_true") # デフォルトFalse # データセットが小さくなり(未実装)、epochが1になる。
    argparser.add_argument('--without_neptune',action="store_true") # デフォルトFalse
    argparser.add_argument('--use_model_iter',action="store_true") # デフォルトFalse
    
    argparser.add_argument('--use_in_model_da_times',type=int,default=None) # 1は拡張なし
    argparser.add_argument('--ce_loss_weight',type=float,default=None)
    argparser.add_argument('--pair_loss_weight',type=float,default=None)
    argparser.add_argument('--sent_loss_weight',type=float,default=None)
    argparser.add_argument('--ce_use_emb',action="store_true") # デフォルトはFalse
    argparser.add_argument('--sent_only_use_cls',action="store_true") # デフォルトはFalse

    argparser.add_argument('--cl_loss_weight',type=float,default=None)
    argparser.add_argument('--pseudo_neg',type=int,default=None)

    args=argparser.parse_args()
    args=vars(args) # args を辞書に変換
    args.pop("execute_mode")
    temp={k:v for k,v in args.items()}
    for k,v in args.items():
        if v==None:
            print(f"Debug pop args {k}")
            temp.pop(k)
    args=temp
    experiment_all_year(write_only=write_only,**args)

def experiment_all_year(write_only,base_dir,meta_name,architecture,model_path,augmentation_flag=False,loop_time=5,model_remove=False,neptune_tags=[],debug=False,without_neptune=False,use_model_iter=False,**args):
    """
    モデルを変えないで全ての年度を一つずつTestデータとして trian.controllerを制御するための関数
    """
    # pprint.pprint(sys.path)
    base_dir=os.path.join(base_dir,meta_name)
    # os.mkdir(base_dir)
    files=DATA_FILE_PATHS

    start=0 
    if debug:
        end=2
        # end=len(files)
    else:
        end=len(files)

    if debug:
        loop_time=2
        neptune_tags.append("Debug")
    
    nep_create_flag=False
    if not without_neptune:
        nep=create_neptune_run_instance(meta_name,tags=neptune_tags)
        nep_create_flag=True
    else:
        nep=None
    # neptune インスタンスを全てまとめて作った場合、Logが足りなくなるときは、controller内部で作成する。
    # nep=None

    log_level=10 if debug else 20
    logger=create_logger("all_year",log_level=log_level)
    logger.info(f"start all year name : {meta_name} !\n\n")
    for i in range(start,end):
        # i ... 省くindex
        dir_name=f"without_{os.path.splitext(os.path.basename(files[i]))[0]}"
        logger.info(f"start one {meta_name} / {dir_name}")

        try:
            os.mkdir(os.path.join(base_dir,dir_name)) # 実行スクリプトからの相対パス
        except Exception as e:
            logger.error(e)
            logger.error(os.getcwd())
            logger.error(os.listdir())
            print(e)
            print(os.getcwd())
            print(os.listdir())
            raise e



        train_path=os.path.join(base_dir,dir_name,"train.tsv") # 実行スクリプトからの相対パス
        create_without_data(files, i, train_path)
        test_path=os.path.join(DATA_DIR_PATH,files[i])
        controller_args={
            "train_file_path":train_path,"arch":architecture,
            "name":dir_name,"base_dir":base_dir,
            "loop_time":loop_time,"use_model_iter":use_model_iter,"model_remove":model_remove,
            "pretrained_model_name_or_path":model_path,
            "test_path":test_path,"aug_path": None if not augmentation_flag else os.path.join(AUGMENTATION_FILE_PATH),
            "without_neptune":without_neptune,"neptune_instance":nep,"neptune_init_tags":[],
            "jlbert_token":None,"debug":debug,"logger":logger
        }
        for key in args:
            controller_args[key]=args[key]
        if not write_only:
            controller(**controller_args)
            # controller(train_file_path=train_path,arch=architecture,
            #     name=dir_name,base_dir=base_dir,
            #     loop_time=loop_time,use_model_iter=use_model_iter,model_remove=model_remove,
            #     pretrained_model_name_or_path=model_path,
            #     test_path=test_path,aug_path= None if not augmentation_flag else os.path.join(AUGMENTATION_FILE_PATH),
            #     without_neptune=without_neptune,neptune_instance=nep,neptune_init_tags=[],
            #     jlbert_token=None,debug=debug,logger=logger,**args
            #     )
        else:
            args_str=json.dump(controller_args) # indent=2,
            with open("hoge.sh","w") as f:
                f.write(f"python train.py {args_str}")

            
            
        # Attention !!
        # default batch_size:int=12,epochs:int=5,max_len:int=256,lr:float=1e-5,jlbert=None
        # augmentation_dir ... default

        logger.info(f"end one year {meta_name} / {dir_name} \n\n")

    # if nep_create_flag:
        # nep.stop()
    logger.info(f"end all year name : {meta_name} !")

def write_sh_script(output_dir="results",debug=False,without_neptune=False):
    # ファイルは output_dir/meta_name(arch_model-name_aug-name)/command.shに作成される。
    # このスクリプトは任意の場所から呼び出して良い。そのとき、相対パスを記述する時はその任意の場所基準で記述をする。実行においては、その場所から実行する必要がある。
    
    # ---------- 以下必要に応じて書き換える ----------
    # モデルの名前は {architectures.key}_{models.key}_{data_augs.key}となる。必要に応じてkeyでは"-"を使用して詳細に記述せよ。
    architectures={
        # "plain":[{}],
        "MSCL":[{"use_in_model_da_times":1,"ce_loss_weight":1.0,"pair_loss_weight":0.02,"sent_loss_weight":1.0},
                # {"use_in_model_da_times":1,"ce_loss_weight":1.0,"pair_loss_weight":1.0,"sent_loss_weight":1.0}
            ]
    } 
    models={
        # "Pretrained_Model_Name":{"path":"","use_model_iter":False,"in_huggingface":True},
        "tohoku-bert":{"path":"cl-tohoku/bert-base-japanese-whole-word-masking","use_model_iter":False,"in_huggingface":True},
    }
    data_augs={
        "DABase":True,
        "NA":False,
        # "MYDA":"path", # TODO str path not implemented
    }
    # ---------- ここまで----------

    def write_sh(sh_path,python_file_path,meta_name,arch,model_name,model_path,aug_name,is_aug,debug,without_neptune,use_model_iter,arch_settings):
        with open(sh_path,"w") as f:
            if True:
                sh= ""
                sh+= "cd `dirname $0`\n"
                sh+= f"python3 {python_file_path} exec"
                # sh+=f" \\\n\t--base_dir {os.path.join('../',base_dir)}"
                sh+=f" \\\n\t--base_dir ../" # どの下に実行結果を作るか。通常 meta_name/command.shとなるために, ..にしておく。
                sh+=f" \\\n\t--meta_name {meta_name}"
                sh+=f" \\\n\t--architecture {arch}"
                sh+=f" \\\n\t--model_path {model_path}"
                sh+=f" \\\n\t--neptune_tag H18-R01 {meta_name} {arch} {model_name} {aug_name}"
                sh+= " \\\n\t--model_remove" # Attention　model_removeはいろいろなところでFalseであるが、ここではTrue
            if is_aug:
                sh+= " \\\n\t--augmentation_flag"
            if debug:
                sh+= " \\\n\t--debug"
            if without_neptune:
                sh+= " \\\n\t--without_neptune"
            if use_model_iter:
                sh+= " \\\n\t--use_model_iter"
            for key,value in arch_settings.items():
                sh+= f" \\\n\t--{key} {value}"
            if False:
                # TODO 例えば architectureがdoubleの時のパラメタ共有など。
                # 新しいパラメタを追加するには、do_experimentでargparserに追加が必要である。
                # また、train.pyにおいて辞書args から該当パラメタを取り出し、適切に処理を実行する必要がある。
                sh+=""
            if True:
                sh+= "\n# this script is create from command.py\n"
            f.write(sh)
            os.chmod(sh_path,0o755)
            print("write ",sh_path)

    for arch in architectures.keys():
        for arch_settings in architectures[arch]:
            for model_name in models.keys():
                model_path=models[model_name]["path"]
                use_model_iter=models[model_name]["use_model_iter"]
                in_huggingface=models[model_name]["in_huggingface"]
                for aug_name,is_aug in data_augs.items():
                    base_dir=output_dir
                    meta_name=f"{arch}_{model_name}_{aug_name}"
                    create_dir=os.path.join(base_dir,meta_name)
                    os.mkdir(create_dir)
                    sh_path=os.path.join(create_dir,"command.sh")
                    python_file_path=os.path.abspath(__file__)
                    # model_path=model_path if in_huggingface else os.path.join('../',model_path)
                    write_sh(sh_path,python_file_path,meta_name,arch,model_name,model_path,aug_name,is_aug,debug,without_neptune,use_model_iter,arch_settings)




# python command.py
if __name__ =="__main__":
    if sys.argv[1]=="exec":
        do_experiment_all_year_from_shell()# debug,without_neptune
        print("Warning controoler may need to do from sh . so Implemnt Command.py")
        # COntroller よりtrain.pyの方がshで動かしたいかも。
        # do_experiment_all_year_from_shell(write_only=True)# write_onlyはcontrollreを動かすshを出力する
    elif sys.argv[1]=="write":
        write_sh_script()
    else:
        raise Exception("Incorrect argv[1]. set `exec` or `write`")

