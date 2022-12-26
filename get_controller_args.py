import argparse
import sys
import json

# TODO ADD help
def get_controller_args(func=None):
    # func ... func(argparser):内部で独自コマンド追加

    # 繰り返す場合 lで指定する。なくても良い。
    # 出力dirは、[BASE_DIR]/[NAME]/[I:繰り返す場合のみ]/ である。
    # その配下にモデルなど全て置かれる。
    # 入力データ、Testは必須。入力の一部はValidになる。入力データ、テストデータはBASE_DIRなどは使用しない。
    # データの形式は、Hypothesis,Premise,Extentionの形式。Extentionはなくても良いようにしたい。

    # セーブディレクトリについて
    # base_dir/name/iter/ に保存される。
    # loop_timeが1の時のiterはその層はスキップされる。

    # neptune上の扱いについて
    # 1. neptune のnameは、nameが使われる。ただし、neptune_instanceが与えられた時は使われない。
    # 2. neptune上の保存場所について
    # 全て同一のRunとして保存される。
    # name/にparametersが保存される。
    # 学習ごとの結果は、name/iter に保存される。
    # 3. 複数のnameをneptuneで一つのrunとしてまとめたい(例えば年度を変えて交差検証的な)場合、
    # controllerに対して、通常の引数として、pythonプログラムからneptune_instanceに
    # 、init後のinstanceを与えると上手にやってくれる。終わったら外でstopをする必要がある。

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_file_path', type=str,required=True)
    argparser.add_argument('--arch', type=str,required=True)
    argparser.add_argument('--name', type=str,required=True)
    argparser.add_argument('--base_dir', type=str)
    argparser.add_argument('--loop_time',type=int)
    argparser.add_argument('--use_model_iter',action="store_true") # default false # これがtの時、modelの名前は[model_path]_[loop_iter]となる。loop_iterは1始まりで、loop_timeが1の時も_1がつく。
    argparser.add_argument('--model_remove',action="store_true") # デフォルトFalse # old -r
    argparser.add_argument('--pretrained_model_or_path',type=str)
    argparser.add_argument('--test_path', type=str)
    argparser.add_argument('--aug_path', type=str)
    argparser.add_argument('--batch_size',type=int)
    argparser.add_argument('--epochs',type=int)
    argparser.add_argument('--max_len',type=int)
    argparser.add_argument('--lr',type=float)
    argparser.add_argument('--without_neptune',action="store_true") # デフォルトFalse
    argparser.add_argument('--neptune_init_tags',nargs="*") # 空白で複数
    argparser.add_argument('--jlbert_token',type=str) # JL Bertを使う時、access tokenをセットする。この時、tokenizerはBertTokenizerを使用する。 # old lic_token
    argparser.add_argument('--debug',action="store_true") # デフォルトFalse # データセットが小さくなり(未実装)、epochが1になる。
    if func:
        func(argparser)
    return argparser.parse_args()

def str2dict_args_from_sh():
    #　第一引数は文字列の辞書である。つまり、 python test.py '{"start": "2004-1-10", "end": "2004-8-31"}' のような感じ。
    argvs = json.loads(sys.argv[1])
    return argvs
