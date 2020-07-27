import torch.nn.functional as F 
import torch 
import torch.nn as nn 
import numpy as np 
from utils import load_config


config_file = 'config.yaml'
config = load_config(config_file)
labels = config['PARAMETERS']['labels']

def to_one_hot(seg, labels=labels):

    assert labels is not None
    assert isinstance(labels, list), 'lables must be in list and cannot be empty'

    if len(seg.shape) == 4:
        dim = 1
    elif len(seg.shape) == 5:
        dim = 2
    else:
        assert len(seg.shape) in (4,5), 'segmentation dimenstion must be 4 (NxDxHxW) of 5 (NxTxDxHxW)'

    seg = seg.cpu().unsqueeze(dim).long()
    n_channels = len(labels)
    # n_channels = 2
    shape = np.array(seg.shape)
    shape[dim] = n_channels
    one_hot_seg = torch.zeros(tuple(shape), dtype=torch.long)

    one_hot_seg = one_hot_seg.scatter_(dim, seg, 1) #dim, index, src

    return one_hot_seg

class GneralizedDiceLoss(nn.Module):
    def __init__(self, labels, epsilon=1e-5):
        super(GneralizedDiceLoss, self).__init__()

        self.epsilon = epsilon
        self.labels = labels

    def forward(self, logits, targets):

        targets = to_one_hot(targets, self.labels).to(logits.device)
        targets.requires_grad = False
    
        channels = logits.shape[1]
        logits = logits.permute(1, 0, 2, 3, 4)
        logits = logits.contiguous().view(channels, -1)
        targets = targets.permute(1, 0, 2, 3, 4)
        targets = targets.contiguous().view(channels, -1).type_as(logits)

        intersections = logits * targets
        unions = logits + targets
        weights = 1. / (F.log_softmax(targets.sum(-1).type_as(logits), dim=-1) + self.epsilon)
        dice = torch.sum(weights * intersections.sum(-1)) / torch.sum(weights * unions.sum(-1)) 

        return 1 - 2 * dice


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, labels, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()

        self.labels = labels
        self.reduction = reduction
        self.bce = nn.BCELoss(reduction=self.reduction)
        self._epsilon = 1e-5

    def forward(self, logits, targets):
        targets = to_one_hot(targets, self.labels).to(logits.device)
        targets.requires_grad = False

        if len(logits.shape) == 6:
            N, T, C, Z, Y, X = logits.shape
            logits = logits.contiguous().view(-1, C, Z, Y, X)
            targets = targets.contiguous().view(-1, C, Z, Y, X)

        channels = logits.shape[1]
        logits = logits.permute(1, 0, 2, 3, 4)
        logits = logits.contiguous().view(channels, -1)
        targets = targets.permute(1, 0, 2, 3, 4)
        targets = targets.contiguous().view(channels, -1).type_as(logits)

        weights = 1. / (F.log_softmax(targets.sum(-1).type_as(logits), dim=-1) + self._epsilon)
        # loss = torch.zeros_like(weights)
        loss = 0
        for i in range(channels):
            loss += self.bce(logits[i], targets[i])

        return loss / channels

        

class DiceLoss(nn.Module):
    def __init__(self, labels, ignore_idx=None, epsilon=1e-5):
        super(DiceLoss, self).__init__()

        # smooth factor
        self.epsilon = epsilon
        self.labels = labels
        self.ignore_idx = ignore_idx
        if isinstance(self.ignore_idx, int):
            self.ignore_idx = [self.ignore_idx]

    def forward(self, logits, targets):

        targets = to_one_hot(targets, self.labels).to(logits.device)
        targets.requires_grad = False

        if len(logits.shape) == 6:
            N, T, C, Z, Y, X = logits.shape
            logits = logits.contiguous().view(-1, C, Z, Y, X)
            targets = targets.contiguous().view(-1, C, Z, Y, X)

        batch_size = targets.size(0)
        tot_loss = 0
        count = 0 
        
        for i in range(logits.shape[1]):
            if self.ignore_idx and i not in self.ignore_idx:
                logit = logits[:, i].view(batch_size, -1).type(torch.FloatTensor)
                target = targets[:, i].view(batch_size, -1).type(torch.FloatTensor)
                intersection = (logit * target).sum(-1)
                dice_score = 2. * intersection / ((logit + target).sum(-1) + self.epsilon)
                loss = torch.mean(1. - dice_score)
                tot_loss += loss
                count += 1 
            elif not self.ignore_idx:
                logit = logits[:, i].view(batch_size, -1).type(torch.FloatTensor)
                target = targets[:, i].view(batch_size, -1).type(torch.FloatTensor)
                intersection = (logit * target).sum(-1)
                dice_score = 2. * intersection / ((logit + target).sum(-1) + self.epsilon)
                loss = torch.mean(1. - dice_score)
                tot_loss += loss
                count += 1 
                         
        return tot_loss / count


class DiceLossLSTM(DiceLoss):
    def __init__(self, labels, epsilon=1e-05):
        DiceLoss.__init__(self, labels, epsilon=1e-05)

    def forward(self, logits, targets):

        targets = to_one_hot(targets, self.labels).to(logits.device)
        targets.requires_grad = False

        batch_size, timestep, channels, _, _, _ = logits.shape
        tot_loss = 0

        for tp in range(timestep):
            logit = logits[:, tp, ...]
            target = targets[:, tp, ...]

            timestep_loss = 0
            for i in range(channels):
                pred = logit[:, i].view(batch_size, -1).type(torch.FloatTensor)
                seg = target[:, i].view(batch_size, -1).type(torch.FloatTensor)
                intersection = (pred*seg).sum(-1)
                dice_score = 2. * intersection / ((pred + seg).sum(-1) + self.epsilon)
                loss = torch.mean(1. - dice_score)
    
                timestep_loss += loss
            
            timestep_loss /= channels
            tot_loss += timestep_loss

        return tot_loss/timestep


class BCELossLSTM(nn.Module):
    def __init__(self, labels):
        super(BCELossLSTM, self).__init__()
        self.labels = labels

    def forward(self, logits, targets):
        targets = to_one_hot(targets, self.labels).to(logits.device)
        targets.requires_grad = False

        loss_f = nn.BCELoss()
        loss = loss_f(logits, targets.float())

        return loss


if __name__ == "__main__":

    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    yp = np.random.random(size=(2, 5, 3, 3, 3))
    yp = torch.from_numpy(yp)
    
    yt = torch.ones(size=(2, 3, 3, 3))
    # print(yp)
    dl1 = BinaryDiceLoss(labels=[0, 1, 2, 3, 4], index=4)
    # dl = DiceLossLSTM(labels=[0, 1, 2, 3, 4])
    # dl = BCELossLSTM(labels=[0, 1, 2, 3, 4])
    # dl2 = GneralizedDiceLoss(labels=[0, 1, 2, 3, 4])
    # dl = WeightedCrossEntropyLoss(labels=[0, 1, 2, 3, 4])
    print(dl1(yp, yt).item())
    # print(dl(yp, yt).item())
    

    
    # bilstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, bidirectional=True)
    # input = torch.randn(5, 3, 10)
    # h0 = torch.randn(4, 3, 20)
    # c0 = torch.randn(4, 3, 20)
    # output, (hn, cn) = bilstm(input, (h0, c0))
    # print(np.unique(output.detach().cpu()))
    



    
    
