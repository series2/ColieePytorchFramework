# Env File
ルートに.envファイルを設置せよ。
中身の例

```
NEPTUNE_PROJECT=[NEPTUNE_PROJECT]
NEPTUNE_API_TOKEN=[NEPTUNE_API_TOKEN]
DATA_DIR_PATH=[DIR_PATH]
AUGMENTATION_FILE_PATH=[path].tsv
```


# 学習用シェルスクリプト
また、学習用ファイルの例を以下に示す。
## train.pyで固定データを使って学習する例
```
python train.py \
    --train_file_path "train.tsv" \
    --arch "plain" \
    --name "FixedData" \
    --base_dir "results/" \
    --loop_time 10 \
    --model_remove \
    --pretrained_model_or_path "cl-tohoku/bert-base-japanese-whole-word-masking" \
    --test_path "test.tsv" \
    --aug_path "aug.tsv" \
    --valid_path "valid.tsv" \
    --reuse_nep \
    --neptune_init_tags "plain" "tohoku-bert" "DABase" "plain_tohoku-bert_DABase" "FixedData"
```

# command.pyを使用してDATA_DIR_PATHにあるデータで交差検証的学習する例
ただし、command.pyは推奨されません。代わりに次のコマンドが推奨されます。
```
cd `dirname $0`
python3 /work/source/PytorchFramework/command.py exec \
	--base_dir ../ \
	--meta_name plain_tohoku-bert_DABase \
	--architecture plain \
	--model_path cl-tohoku/bert-base-japanese-whole-word-masking \
	--neptune_tag plain_tohoku-bert_DABase plain tohoku-bert DABase \
	--model_remove \
	--augmentation_flag
# this script is create from command.py
```

# train.pyを使用してDATA_DIR_PATHにあるデータで交差検証的学習する例
推奨
シェルスクリプトを容易に分割できるので、複数プロセスでの実行の分割が楽です。
また、Pythonプロセスを一回一回切るので、なぜか残るGPUのメモリをリセットできます。
複数プロセスで実行をする場合、同一のインスタンスに対する学習に対するロックが甘いので気をつけてください。
```
for year in "H18" "H19" "H20" "H21" "H22" "H23" "H24" "H25" "H26" "H27" "H28" "H29" "H30" "R01"
do
    cd `dirname $0` # forのreset
    mkdir "without_"$year"_jp"
    cd "without_"$year"_jp"
    for trial in 1 2 3 4 5
    do
    mkdir $trial
    cd $trial
    if [ ! -e "test_CF.tsv" ]; then
            # ファイルが存在しない場合のみ実行する
        python train.py \
            --train_file_path "without_"$year".tsv" \
            --arch "MSCL" \
            --name "without_"$year"_jp/"$trial \
            --base_dir "../../" \
            --loop_time 1 \
            --model_remove \
            --pretrained_model_or_path "cl-tohoku/bert-base-japanese-whole-word-masking" \
            --test_path "/tsv/"$year"_jp.tsv" \
            --aug_path "/tsv/coliee_2019.tsv" \
            --use_in_model_da_times 2 \
            --ce_loss_weight 1.0 \
            --pair_loss_weight 0.1 \
            --sent_loss_weight 0.1 \
            --reuse_nep \
            --neptune_init_tags MSCL_tohoku-bert_DABase MSCL tohoku-bert DABase
            # --debug
    fi
    cd ../
    done
    cd ../ # forのreset
done
```