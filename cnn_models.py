import torch

class Test_CNN_5(torch.nn.Module):
  def __init__(self, n_classes, window_len=3, encoding_dim=20, feature_map_dim=256, dropout=0.3):
    super().__init__()
    self.layer1 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = encoding_dim, out_channels = feature_map_dim, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer2 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer3 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer4 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer5 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.fc =   torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = n_classes, kernel_size=1)

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.fc(x)
    return x 


class Test_CNN_6(torch.nn.Module):
  def __init__(self, n_classes, window_len=3, encoding_dim=20, feature_map_dim=256, dropout=0.3):
    super().__init__()
    self.layer1 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = encoding_dim, out_channels = feature_map_dim, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer2 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer3 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer4 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer5 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer6 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.fc =   torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = n_classes, kernel_size=1)

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)
    x = self.fc(x)
    return x 


class Test_CNN_7(torch.nn.Module):
  def __init__(self, n_classes, window_len=3, encoding_dim=20, feature_map_dim=256, dropout=0.3):
    super().__init__()
    self.layer1 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = encoding_dim, out_channels = feature_map_dim, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer2 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer3 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer4 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer5 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer6 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer7 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.fc =   torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = n_classes, kernel_size=1)

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)
    x = self.layer7(x)
    x = self.fc(x)
    return x 


class Dil_CNN_5(torch.nn.Module):
  def __init__(self, n_classes, window_len=3, encoding_dim=20, feature_map_dim=256, dropout=0.3):
    super().__init__()
    self.layer1 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = encoding_dim, out_channels = feature_map_dim, 
                      dilation=1, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer2 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, 
                      dilation=2, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer3 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, 
                      dilation=4, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer4 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, 
                      dilation=8, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer5 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, 
                      dilation=16, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.fc =   torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = n_classes, kernel_size=1)

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.fc(x)
    return x 


 class Dil_CNN_6(torch.nn.Module):
  def __init__(self, n_classes, window_len=3, encoding_dim=20, feature_map_dim=256, dropout=0.3):
    super().__init__()
    self.layer1 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = encoding_dim, out_channels = feature_map_dim, 
                      dilation=1, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer2 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, 
                      dilation=2, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer3 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, 
                      dilation=4, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer4 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, 
                      dilation=8, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer5 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, 
                      dilation=16, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer6 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, 
                      dilation=32, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.fc =   torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = n_classes, kernel_size=1)

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)
    x = self.fc(x)
    return x 


class Dil_CNN_7(torch.nn.Module):
  def __init__(self, n_classes, window_len=3, encoding_dim=20, feature_map_dim=256, dropout=0.3):
    super().__init__()
    self.layer1 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = encoding_dim, out_channels = feature_map_dim, 
                      dilation=1, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer2 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, 
                      dilation=2, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer3 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, 
                      dilation=4, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer4 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, 
                      dilation=8, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer5 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, 
                      dilation=16, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer6 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, 
                      dilation=32, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.layer7 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = feature_map_dim, 
                      dilation=64, kernel_size=window_len, padding='same'),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.fc =   torch.nn.Conv1d(in_channels = feature_map_dim, out_channels = n_classes, kernel_size=1)

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)
    x = self.layer7(x)
    x = self.fc(x)
    return x 


class CARP_CNN(torch.nn.Module):
  def __init__(self, n_classes, window_len=3, encoding_dim=1024, dropout=0.3):
    super().__init__()
    self.layer1 = torch.nn.Sequential(
      torch.nn.Conv1d(in_channels = encoding_dim, out_channels = 64, kernel_size=1),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout)
    )
    self.fc =   torch.nn.Conv1d(in_channels = 64, out_channels = n_classes, kernel_size=1)

  def forward(self, x):
    x = self.layer1(x)
    x = self.fc(x)
    return x 