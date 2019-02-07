import torch
from progress.bar import Bar

from optim_latent import initLatent
from datasets.Fusion import unpack_splitted
from layers.ShapeConsistencyCriterion import ShapeConsistencyCriterion
from utils.utils import AverageMeter
import ref

def forward_dataset(model, loader, epoch, max_epoch):
      nViews = ref.nViews
      model.eval()
      loss_mean = AverageMeter()
      bar = Bar('DIAL forward:', max=len(loader))
      for i, (data, label, meta) in enumerate(loader):
            print "data size: " + str(data.size())
            print "label size: " + str(label.size())
            print "meta size: " + str(meta.size())
            data_var = torch.autograd.Variable(data.cuda())
            label_var = torch.autograd.Variable(label)
            output = model(data_var).detach()
            loss = ShapeConsistencyCriterion(nViews, supWeight = 1, unSupWeight = 0, M = None)(output.cpu(), label_var, torch.autograd.Variable(meta))
            print "data: " + str(loss.data[0])
            loss_mean.update(loss.data[0], data.size(0))
            bar.suffix = '[%d / %d] Loss: %.6f' % (epoch, max_epoch, loss_mean.avg)
            bar.next()
      bar.finish()
      return loss_mean

def train_statistics(dial_model, dataset_loader, epochs, source_domain=False):
      dial_model.eval()
      dial_model.set_domain(source=source_domain)
      loss_history = []
      for e in range(epochs):
           loss = forward_dataset(dial_model, dataset_loader, e, epochs)
           loss_history.append(loss)
      return loss_history
