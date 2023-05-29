# faster-rcnn
# train
1.Install dependencies
install PyTorch with GPU
install visdom scikit-image tqdm fire ipdb pprint matplotlib torchnet
2.prepare your dataset
modify voc_data_dir cfg item in utils/config.py
modify (if not voc)
3.(Optional)Prepare caffe-pretrained vgg16

4.Start visdom
nohup python -m visdom.server &
5.
# test
