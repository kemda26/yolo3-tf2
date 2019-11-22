pip install -r requirements.txt
cd datasets && curl http://ufldl.stanford.edu/housenumbers/train.tar.gz -o train.tar.gz && tar -xzf train.tar.gz && rm -rf train.tar.gz
cd datasets && curl http://ufldl.stanford.edu/housenumbers/test.tar.gz -o test.tar.gz && tar -xzf test.tar.gz && rm -rf test.tar.gz
cd datasets/test && rm -rf see_bboxes.m digitStruct.mat && ls -A | wc -l
cd datasets/train && rm -rf see_bboxes.m digitStruct.mat && ls -A | wc -l