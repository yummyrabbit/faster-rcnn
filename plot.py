from __future__ import absolute_import
import matplotlib
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
import resource
from data.util import read_image
import torch as t
from utils.config import opt

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')
# load model
faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
best_path = r'pretrained_model/fasterrcnn.pth'
trainer.load(best_path)
opt.caffe_pretrain = True


# plot proposal boxes
def plotpro():
    l=['bike.jpg','bird.jpg','car.jpg','tv.png']
    for i in range(4):
        path=l[i]
        img = read_image(path)
        _bboxes = trainer.faster_rcnn.pro([img], visualize=True)
        probox = visdom_bbox(img,
                         at.tonumpy(_bboxes[0]))
        trainer.vis.img('proposal boxes'+str(i+1), probox)


# plot predicti boxes
def plotpre():
    l=['bike.jpg','bird.jpg','car.jpg','tv.png']
    for i in range(4):
        path=l[i]
        img = read_image(path)
        _bboxes, _labels, _scores = trainer.faster_rcnn.predict([img], visualize=True)
        pred_img = visdom_bbox(img,
                           at.tonumpy(_bboxes[0]),
                           at.tonumpy(_labels[0]).reshape(-1),
                           at.tonumpy(_scores[0]))
        trainer.vis.img('pred_img'+str(i+1), pred_img)


if __name__ == '__main__':
    import fire

    fire.Fire()
