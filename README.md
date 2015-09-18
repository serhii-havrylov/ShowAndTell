"For millions of years mankind lived just like the animals. Then something happened which unleashed the power of our imagination: we learned to talk".

This project reproduces the model from [Show and Tell: A Neural Image Caption Generator](http://arxiv.org/abs/1411.4555)

Image features are the outputs of the `relu7` layer from the VGG network which you can download [here](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md).

You can download prepared training and validation data from my [google drive](https://drive.google.com/folderview?id=0B-bMt9sukkEhfnlhbm9fczhsTWVQcm1yNkpfVExvU19jWmw0bzl1ZTQ1eDVyeW82Vi1pQ1E&usp=sharing) or you can reproduce image/text feature extraction pipeline as following:

1. Download datasets
    - [MSCOCO](http://mscoco.org/dataset/#download)
    - [Flickr8k](https://illinois.edu/fb/sec/1713398) You should send request for data receiving
    - [Flickr30k](https://illinois.edu/fb/sec/229675) You should send request for data receiving
2. Run python scripts for generating files which store the image paths and corresponding captions
    - run `data_preparation/flickr/flickr8k/build_image_text_match.py`
    - run `data_preparation/flickr/flickr30k/build_image_text_match.py`
    - run `data_preparation/mscoco/build_image_text_match.py`
3. Run python scripts for generating files which store image features
    - run `data_preparation/flickr/extract_features.py`
    - run `data_preparation/mscoco/extract_features.py`
4. Run python scripts for generating training and validation data
    - run `data_preparation/merge_all_data.py`

To train model run `caption_generation_model/train.py` or you can download pretrained model from my [google drive](https://drive.google.com/folderview?id=0B-bMt9sukkEhfnlhbm9fczhsTWVQcm1yNkpfVExvU19jWmw0bzl1ZTQ1eDVyeW82Vi1pQ1E&usp=sharing)

If you want to use the pretrained model run minimalistic flask app `caption_generation_server/app.py` (Note: it requires installed [caffe](https://github.com/BVLC/caffe) and its python interface `pycaffe`)
