


# eta = torch.randn(images.shape)
# eta = eta.uniform_(-eps,eps)

# adv_img = torch.clamp(image, 0.0, 1.0)

# signed_grad = images.grad.sign()
# adv_img -= self.alpha() * signed_grad
# eta = torch.clamp(adv_img - images, 0.0, 1.0)
# adv_img = torch.clamp(images + eta, 0.0, 1.0)

import torch
from torch import nn
from copy import deepcopy
from .base import Attacker
from torch.cuda import amp
from utils.utils import Empty
import numpy as np
import torchvision
class TOG_vanishing(Attacker):
    def __init__(self, model, img_transform=(lambda x:x, lambda x:x), use_amp=False):
        super().__init__(model, img_transform)
        self.use_amp=use_amp
        self.call_back=None
        self.img_loader=None

        if use_amp:
            self.scaler = amp.GradScaler()

    def set_para(self, eps=8, alpha=lambda:2, iters=10, **kwargs):
        super().set_para(eps=eps, alpha=alpha, iters=iters, **kwargs)

    def set_call_back(self, call_back):
        self.call_back=call_back

    def set_img_loader(self, img_loader):
        self.img_loader=img_loader

    #add function
    def init_adv_img(self, images):
        eta = torch.randn(images.shape)
        eta = eta.uniform_(-self.eps, self.eps)
        adv_img = torch.clamp(images + eta, 0, 255) 
        return adv_img

    def step(self, images, labels, loss):
        
        #print(labels.size())
        #print((np.array(labels)).shape)
        #print(labels) # n * 5
        #input()
        #labels[0][:,-1] = 0
        with amp.autocast() if self.use_amp else Empty():
            images.requires_grad = True
            outputs = self.model(images)

            self.model.zero_grad()
            cost = loss(outputs, labels)

        if self.use_amp:
            self.scaler.scale(cost).backward()
        else:
            cost.backward()

        # adv_images = (images + self.alpha() * images.grad.sign()).detach_()
        # eta = torch.clamp(adv_images - self.ori_images, min=-self.eps, max=self.eps)

        # fix
        # signed_grad = images.grad.sign() # this 'images' is the init_adv_img
        # adv_images = (images - self.alpha() * signed_grad()).detach_()
        adv_images = (images - self.alpha() * images.grad.sign()).detach_()
        eta = torch.clamp(adv_images - self.ori_images, min=-self.eps, max=self.eps)

        images = self.img_transform[0](torch.clamp(self.img_transform[1](self.ori_images + eta), min=0, max=255).detach_())

        return images

    def set_data(self, images, labels):
        self.ori_images = deepcopy(images)
        self.images = images
        self.labels = labels

    def __iter__(self):
        self.atk_step=0
        return self

    def __next__(self):
        self.atk_step += 1
        if self.atk_step>self.iters:
            raise StopIteration

        with self.model.no_sync() if isinstance(self.model, nn.parallel.DistributedDataParallel) else Empty():
            self.model.eval()

            self.images = self.forward(self, self.images, self.labels)

            self.model.zero_grad()
            self.model.train()

        return self.ori_images, self.images.detach(), self.labels

    def attack(self, images, labels):
        self.ori_images = deepcopy(images)
       # print(images.shape)
       # input()
        #init_adv_img(add)
        images = self.init_adv_img(images)
        # print(labels)
        # input()
        labels[0][:,-1] = 0
        #print(labels)
        #input()

        for i in range(self.iters):
            self.model.eval()

            images = self.forward(self, images, labels)

            self.model.zero_grad()
            self.model.train()
            if self.call_back:
                self.call_back(self.ori_images, images.detach(), labels)

        #torchvision.utils.save_image(images.data,"togvanishing_adv.jpg")
        #input()
        return images
