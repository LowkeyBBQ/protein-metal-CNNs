from torchvision.ops import focal_loss

class FocalLoss:
  def __init__(self, alpha = 0.25, gamma=2):
    self.alpha = alpha
    self.gamma = gamma
  def __call__(self, prefds, targets):
    return focal_loss.sigmoid_focal_loss(preds, targets, 
                                         self.alpha, self.gamma)