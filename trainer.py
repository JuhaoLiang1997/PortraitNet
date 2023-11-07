from focal_loss import FocalLoss
import torch.nn.functional as F
import torch.nn as nn
import torch, shutil, os, logging
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np

class PortraitTrainer(object):
    def __init__(self, args, model, optimizer, train_dataloader, test_dataloader, multiple):
        super().__init__()
        
        # train instances
        self.loss_softmax = nn.CrossEntropyLoss(ignore_index=255)
        self.loss_focalloss = FocalLoss(gamma=2)
        self.loss_kl = lambda student_outputs, teacher_outputs, T: \
            nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * T * T
        
        # train epoch
        self.temperature =  args.portrait.temperature
        self.alpha = args.portrait.alpha
        self.edgeRatio = args.portrait.edgeRatio

        # train
        self.n_epoch = args.train.n_epoch
        self.learning_rate = args.train.learning_rate
        self.learning_rate_step = args.train.learning_rate_step
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.multiple = multiple

        self.printfreq = 1
        self.output_path = args.output_path
        self.device = args.device

    def train(self):
        min_loss = 100000
        for epoch in tqdm(range(self.n_epoch), postfix='Epoch'):
            lr = self.learning_rate * (0.95 ** (epoch // self.learning_rate_step))
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = lr * self.multiple[i]

            train_loss = self.train_epoch()
            # logging.info(next(self.model.parameters())[0][0][0][0])
            test_loss = self.test_epoch()
            tqdm.write(f"test loss: {test_loss}, train loss: {train_loss}, lr: {lr}")
            if test_loss < min_loss:
                min_loss = test_loss
                is_best = True
            else:
                is_best = False
            self.save_checkpoint(
                state={
                    'epoch': epoch+1,
                    'loss': test_loss,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                },
                is_best=is_best,
                root=self.output_path
            )

    def train_epoch(self):
        self.model.train()
        train_loss = 0
        for input_ori, input, edge, mask in self.train_dataloader:
            input_ori_var = Variable(input_ori.to(self.device))
            input_var = Variable(input.to(self.device))
            edge_var = Variable(edge.to(self.device, dtype=torch.int64))
            mask_var = Variable(mask.to(self.device, dtype=torch.long))

            output_mask, output_edge = self.model(input_var)
            loss_mask = self.loss_softmax(output_mask, mask_var)
            
            loss_edge = self.loss_focalloss(output_edge, edge_var) * self.edgeRatio
            # loss = loss_mask + loss_edge

            # Stability
            output_mask_ori, output_edge_ori = self.model(input_ori_var)
            loss_mask_ori = self.loss_softmax(output_mask_ori, mask_var)
            
            loss_edge_ori = self.loss_focalloss(output_edge_ori, edge_var) * self.edgeRatio
            
            loss_stability_mask = self.loss_kl(output_mask, Variable(output_mask_ori.data, requires_grad = False), self.temperature) * self.alpha
            loss_stability_edge = self.loss_kl(output_edge, Variable(output_edge_ori.data, requires_grad = False), self.temperature) * self.alpha * self.edgeRatio
                
            # total loss
            # loss = loss_mask + loss_mask_ori + loss_edge + loss_edge_ori + loss_stability_mask + loss_stability_edge
            # loss = loss_mask + loss_mask_ori + loss_stability_mask + loss_edge
            loss = loss_mask
            train_loss += loss
            # logging.info(f"loss_mask: {loss_mask}, loss_mask_ori: {loss_mask_ori}, loss_stability_mask: {loss_stability_mask}, loss_stability_edge: {loss_stability_edge}, loss_edge: {loss_edge}, loss_edge_ori: {loss_edge_ori}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return train_loss / len(self.train_dataloader)

    def test_epoch(self):
        # switch to eval mode
        self.model.eval()
        iou = 0.0
        softmax = nn.Softmax(dim=1)
        
        with torch.no_grad():
            for input_ori, input, edge, mask in self.test_dataloader:
                # input_ori_var = Variable(input_ori.to(self.device))
                input_var = Variable(input.to(self.device))
                # edge_var = Variable(edge.to(self.device, dtype=torch.int64))
                mask_var = Variable(mask.to(self.device, dtype=torch.long))
                
                output_mask, output_edge = self.model(input_var)
                prob = softmax(output_mask)[:,1,:,:]
                pred = prob.data.cpu().numpy()
                pred[pred>0.5] = 1
                pred[pred<=0.5] = 0
                # logging.info(f"pred: {np.sum(pred>0.5)}/{pred.size}, mask: {torch.sum(mask_var>0.5).item()}/{mask_var.numel()}")
                iou += self.calcIOU(pred, mask_var[0].data.cpu().numpy())
                    
        return 1.0-iou/len(self.test_dataloader)

    def calcIOU(self, img, mask):
        sum1 = img + mask
        sum1[sum1>0] = 1.0
        sum2 = img + mask
        sum2[sum2<2] = 0.0
        sum2[sum2>=2] = 1.0
        if np.sum(sum1) == 0:
            return 1.0
        else:
            return 1.0*np.sum(sum2)/np.sum(sum1)
        
    def save_checkpoint(self, state, is_best, root, filename='checkpoint.pth.tar'):
        torch.save(state, os.path.join(root, filename))
        if is_best:
            shutil.copyfile(os.path.join(root, filename), os.path.join(root, 'model_best.pth.tar'))  

    def save(self, filename="saved"):
        path = os.path.join(self.output_path, filename+".pth")
        torch.save(self.model.state_dict(), path)
        logging.info(f"Saved in {path}")

def main():
    return