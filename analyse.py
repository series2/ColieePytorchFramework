import pandas as pd

# 年数ごとのばらつきを示すためのプログラム
# file ... /work/source/ColieePytorchFramework/5times_forSumup/SM_DefaultEnb_DABase/analysis_by_trial.csv
def show_year_distribute(file):
    df=pd.read_csv(file)
    df["acc"]=(df["NofA"]/df["NofQ"])*100
    result=df.groupby("year").mean()[["acc"]]
    result["std"]=df.groupby("year").std()[["acc"]]
    result["NofQ"]=df.groupby("year").mean()[["NofQ"]]
    print(result)
"""
year acc std NofQ
H18_jp  63.333333  2.324056   36.0
H19_jp  55.675676  5.604444   37.0
H20_jp  61.463415  2.671817   41.0
H21_jp  55.925926  7.567710   54.0
H22_jp  54.042553  9.101808   47.0
H23_jp  60.000000  7.438025   41.0
H24_jp  62.025316  8.724081   79.0
H25_jp  64.666667  7.207249   60.0
H26_jp  49.189189  3.391865   74.0
H27_jp  57.959184  3.707327   49.0
H28_jp  64.897959  1.707469   49.0
H29_jp  57.931034  6.287787   58.0
H30_jp  58.857143  4.672979   70.0
R01_jp  62.702703  0.805790  111.0
"""


if __name__=="__main__":
    file_path="/work/source/ColieePytorchFramework/5times_forSumup/SM_DefaultEnb_DABase/analysis_by_trial.csv"
    show_year_distribute(file_path)
