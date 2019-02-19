import torch
from progress.bar import Bar

from optim_latent import initLatent
from datasets.Fusion import unpack_splitted
from layers.ShapeConsistencyCriterion import ShapeConsistencyCriterion
from utils.utils import AverageMeter
import models.ADDAResNet as adda
import ref
from opts import opts 
import copy
import torch.nn as nn

args = opts().parse()

def forward_dataset(model, old_model, loader, optimizer, epoch, max_epoch, ):
      nViews = ref.nViews
      model.train()
      loss_mean = AverageMeter()
      bar = Bar('ADDA forward:', max=len(loader))
      for i, (data, label, meta) in enumerate(loader):
            domain = (meta[:,0]==1 or meta[:,0]==-1).long()
 
            data_var = torch.autograd.Variable(data.cuda())
            domain_var = torch.autograd.Variable(domain.cuda())

            fe = model(data_var).detach()
            loss = ShapeConsistencyCriterion(nViews, supWeight = 1, unSupWeight = 0, M = None)(output.cpu(), label_var, torch.autograd.Variable(meta))
            #print "loss: " + str(loss.data[0])
            loss_mean.update(loss.data[0], data.size(0))
            bar.suffix = '[%d / %d] Loss: %.6f' % (epoch, max_epoch, loss_mean.avg)
            bar.next()
      bar.finish()
      return loss_mean


def train_tgt(src_encoder, tgt_encoder, critic, src_data_loader, tgt_data_loader, epochs, th=0.8):

    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()

    params_to_optim = list(filter(lambda p: p.requires_grad, tgt_encoder.parameters()))
    optimizer_tgt = torch.optim.SGD(params_to_optim, args.LR*0.1,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

    params_to_optim = list(filter(lambda p: p.requires_grad, critic.parameters())) 
    optimizer_critic = torch.optim.SGD(params_to_optim, args.LR*0.1,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################

    update_discriminator = True
    update_generator = False
    for epoch in range(epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, _, _), (images_tgt, _, _)) in data_zip:

            ###########################
            # 2.1 train discriminator #
            ###########################

            # make images variable
            images_src = torch.autograd.Variable(images_src.cuda())
            images_tgt = torch.autograd.Variable(images_tgt.cuda())


            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(images_src).detach()
            feat_tgt = tgt_encoder(images_tgt).detach()
            feat_concat = torch.cat((feat_src, feat_tgt), 0).detach()

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())

            # prepare real and fake label
            label_src = torch.autograd.Variable(torch.ones(feat_src.size(0)).long().cuda())
            label_tgt = torch.autograd.Variable(torch.zeros(feat_tgt.size(0)).long().cuda())
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
	    if update_discriminator:
            	loss_critic.backward()
		optimizer_critic.step()

            loss_critic = loss_critic.detach()

            # optimize critic
            

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()
	    if acc.data[0]>th and update_discriminator:
		update_discriminator=False
	    elif acc.data[0]<0.3:
		update_discriminator=True

	    
            ############################
            # 2.2 train target encoder #
            ############################

            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_encoder(images_tgt)

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = torch.autograd.Variable(torch.ones(feat_tgt.size(0)).long().cuda())

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
	    if not update_discriminator:
		    loss_tgt.backward()

		    # optimize target encoder
		    optimizer_tgt.step()

            #######################
            # 2.3 print step info #
            #######################
            if ((step + 1) % 2== 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                      .format(epoch + 1,
                              epochs,
                              step + 1,
                              len_data_loader,
                              loss_critic.data[0],
                              loss_tgt.data[0],
                              acc.data[0]))

    return 0.




def train_discriminator(tgt_model, source_loader, target_loader, epochs):
      src_model = copy.deepcopy(tgt_model)

      for name, param in src_model.named_parameters():
      	param.requires_grad = False

      for name, param in tgt_model.named_parameters():
	if 'layer4' in name or 'fc' in name:
      		param.requires_grad = False

      discriminator = adda.get_discriminator()
      loss = train_tgt(src_model, tgt_model, discriminator.cuda(), source_loader, target_loader, epochs)
      return tgt_model
