wget 'https://www.dropbox.com/s/cxn26olqldqsvig/vgg13.model?dl=1' -O model/vgg13.model
wget 'https://www.dropbox.com/s/7y60cyh2nzjakup/vgg16.model?dl=1' -O model/vgg16.model
python3.6 main.py $1
