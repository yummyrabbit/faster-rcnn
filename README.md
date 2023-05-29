# faster-rcnn
This is a simple faster rcnn implement which reaches 0.645 mAP on VOC 2012, not as good as 0.699 mAP in [origin paper](https://arxiv.org/abs/1506.01497)  
But this is a result without tuning hyperparameters, and i will test its performance after fine tuning.  
You can train or test object detection models by follow the guide below.  
If you meet any question, kindly give me your advice.   
## train(only linux gpu)
### 1.Install dependencies
install PyTorch with GPU  
install visdom scikit-image tqdm fire ipdb pprint matplotlib torchnet  
### 2.prepare your dataset 
modify voc_data_dir cfg item in utils/config.py  
if your dataset is not VOC:  
make sure your dataset has structure below(just like VOC):  

    ---your data_dir
    ------Annotations
    ------ImagesSet
         ---------Main
            ---------trainval.txt
            ---------test.txt
    ------JPEGImages

modify VOC_BBOX_LABEL_NAMES in data/voc_dataset.py
### 3.(Optional)Prepare caffe-pretrained vgg16
you can run below to download caffe-pretrained vgg16  

    python misc/convert_caffe_pretrain.py
modify caffe_pretrain_path cfg item in utils/config.py  
If you want to use pretrained model from torchvision, you may skip this step.  
NOTE, caffe pretrained model has shown slight better performance.  
### 4.Start visdom
    nohup python -m visdom.server &
 Then you can see your train loss,test loss and test map on localhost:8097.  
 Similarly you can modify parameters in utils/config.py to use different training strategy.  
 If your want to visualize other text or picture,write your code in train.py.  
### 5.begin training
 run below to begin your train
 
    python train.py train
## test(only linux gpu)
 run below to start visdom(if not before)
 
    nohup python -m visdom.server &
 modify best_path in plot.py  
Then:  
change l in plot.py and run below to visualize predict pictures:
    
    python plot.py plotpre
change l in plot.py and run below to visualize proposal boxes:
    
    python plot.py plotpro
