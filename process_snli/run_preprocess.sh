#chmod +x ./run_preprocess.sh to change permission
# ./run_preprocess.sh to run
python process-snli.py \
--data_folder "/Users/Lisa/Documents/Grad School/DS-GA 1101/data/snli_1.0" \
--out_folder "/Users/Lisa/Documents/Grad School/DS-GA 1101/data/snli_preprocess"


python preprocess.py \
--srcfile "/Users/Lisa/Documents/Grad School/DS-GA 1101/data/snli_preprocess/src-train.txt" \
--targetfile "/Users/Lisa/Documents/Grad School/DS-GA 1101/data/snli_preprocess/targ-train.txt" \
--labelfile "/Users/Lisa/Documents/Grad School/DS-GA 1101/data/snli_preprocess/label-train.txt" \
--srcvalfile "/Users/Lisa/Documents/Grad School/DS-GA 1101/data/snli_preprocess/src-dev.txt" \
--targetvalfile "/Users/Lisa/Documents/Grad School/DS-GA 1101/data/snli_preprocess/targ-dev.txt" \
--labelvalfile "/Users/Lisa/Documents/Grad School/DS-GA 1101/data/snli_preprocess/label-dev.txt" \
--srctestfile "/Users/Lisa/Documents/Grad School/DS-GA 1101/data/snli_preprocess/src-test.txt" \
--targettestfile "/Users/Lisa/Documents/Grad School/DS-GA 1101/data/snli_preprocess/targ-test.txt" \
--labeltestfile "/Users/Lisa/Documents/Grad School/DS-GA 1101/data/snli_preprocess/label-test.txt" \
--outputfile "/Users/Lisa/Documents/Grad School/DS-GA 1101/data/snli_preprocess/" \
--glove "/Users/Lisa/Documents/Grad School/DS-GA 1101/data/glove.6B/glove.6B.300d.txt"

python get_pretrain_vecs.py \
--glove "/Users/Lisa/Documents/Grad School/DS-GA 1101/data/glove.6B/glove.6B.300d.txt" \
--outputfile "/Users/Lisa/Documents/Grad School/DS-GA 1101/data/snli_preprocess/glove.hdf5" \
--dictionary "/Users/Lisa/Documents/Grad School/DS-GA 1101/data/snli_preprocess/.word.dict"