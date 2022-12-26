import pandas as pd
from logging import getLogger,StreamHandler,Formatter
import logging as Logtype
from dotenv import load_dotenv
import neptune.new as neptune
import os,sys

def load_dataset(filename,debug=False):  # validation / testデータをload
    df = pd.read_table(filename, names=('label', 't1', 't2'), na_filter=False)
    if debug:
        df=df.head(50)
    df.fillna('')
    # raise Exception("order of t1 t2 is correct? see utils.py")
    # return list(zip(df.t2, df.t1)), df.label.values.tolist()
    return list(zip(df.t1, df.t2)), df.label.values.tolist()
    # df = pd.read_table(filename,na_filter=False)
    # df.fillna('')
    # if len(df.columns)==4:
    #     df.set_axis(['label', 't1', 't2',"extention"], axis=1)
    #     return list(zip(df.t1, df.t2,df.extention)), df.label.values.tolist()
    # else:
    #     df.set_axis(['label', 't1', 't2'], axis=1)
    #     return list(zip(df.t1, df.t2)), df.label.values.tolist()


def create_logger(name,log_level=Logtype.INFO):
    # log_level 以上のレベルを対象にする。 DEBUG:10,Info:20,Warning:30,Error:40,Critical:50
    logger=getLogger(name)
    logger.setLevel(log_level) # Debugも出力する。
    handler = StreamHandler(sys.stdout)
    handler.setLevel(log_level) 
    logger.addHandler(handler)
    # fmt = Formatter("(%(asctime)s)[%(name)s / %(levelname)s]<l:%(lineno)s> %(message)s") # %(process)d 
    fmt=Formatter('%(asctime)s,%(msecs)03d [%(name)s / %(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
   # ('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')
    handler.setFormatter(fmt)
    # logger.info('Process Start!')
    # logger.debug('debug')
    # logger.warning('warning')
    # logger.error('error')
    return logger

"""
.envファイルフォーマット
NEPTUNE_PROJECT={YOUR_NEPTUNE_PROJECT}
NEPTUNE_API_TOKEN={YOUR_NEPTUNE_API_TOKEN}
DATA_DIR_PATH={YOUR_DATA_DIR_PATH}
AUGMENTATION_FILE_PATH={YOUR_AUGMENTATION_FILE_PATH}
もし存在しなくてもerroorにならないので、必須のDATA_DIR_PATH以外は書かなくても良い。
"""
def create_neptune_run_instance(name,tags=None,nep_only=False):
    load_dotenv(override=True)
    if tags==None:
            tags=[]
    nep=neptune.init_run(
            name=name,
            project=os.getenv('NEPTUNE_PROJECT'),
            api_token=os.getenv('NEPTUNE_API_TOKEN'),
            tags=tags
        )
    if not nep_only:
        print("warning! Neptune Wrapper is not implemented.but neptune is activated!.")
        # nep=NeptuneWrapper(neptune_instance=nep)
    return nep


class NeptuneWrapper():
    def __init__(self,neptune_instance):
        self.nep=neptune_instance
        self.basedir="TODO" # TODO
    def __getitem(self,key):
        # .logで記録できるようなインスタンスを返す。
        self.nep[key]
    def __setitem__(self,key,value):
        self.nep[key]=value
        # TODO ファイル書き込み。保存場所は
    def __delitem__(self,key):
        del self.nep[key]
    def stop(self):
        self.nep.stop()
    # def __getslice__():
        # pass
    # def __setslice__():
        # pass
    # def __delslice__():
        # pass
