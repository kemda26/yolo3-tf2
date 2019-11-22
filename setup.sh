pip install -r requirements.txt
cd datasets 
curl http://ufldl.stanford.edu/housenumbers/train.tar.gz -o train.tar.gz
tar -xzf train.tar.gz && rm -rf train.tar.gz
curl http://ufldl.stanford.edu/housenumbers/test.tar.gz -o test.tar.gz 
tar -xzf test.tar.gz && rm -rf test.tar.gz
rm -rf train/see_bboxes.m train/digitStruct.mat && ls train/ -A | wc -l
rm -rf test/see_bboxes.m test/digitStruct.mat && ls test/ -A | wc -l