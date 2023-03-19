import torch 
import torch.nn as nn
import segmentation_models_pytorch as smp

# Easy pretrained backbone Segmentation network
class UnetModel(nn.Module):
    
    def __init__(self, encoder_name='resnet34', in_channels=3, out_channels_1=1, out_channels_2=1,
                 decoder_channels=(256, 128, 64, 32, 16), doRegression=True):
        
        super(UnetModel, self).__init__()
        
        channel = {'resnet18':512, 'resnet34':512, 'resnet50':2048, 'mobilenet_v2':1280,'efficientnet-b3':384,'dpn68':512}
        self.doRegression = doRegression
        self.Unet = smp.Unet(encoder_name, in_channels=in_channels, classes=1, activation=None,
                        decoder_channels=decoder_channels)
        self.out_1 = nn.Conv2d(decoder_channels[-1], out_channels_1, kernel_size=3, stride=1, padding=1)
        self.out_2 = nn.Conv2d(decoder_channels[-1], out_channels_2, kernel_size=3, stride=1, padding=1)
        
        # Regression model
        if self.doRegression:
            self.out_3 = nn.Conv2d(channel[encoder_name], 8, kernel_size=3, stride=1, padding=1)
            self.mlp = nn.Linear(8*7*7, 8)

    def forward(self, x):
        bs = x.shape[0]
        features = self.Unet.encoder(x)
        if self.doRegression:
            x = self.mlp(self.out_3(features[-1]).view(bs, -1))
            decoder_output = self.Unet.decoder(*features)
            mask_1, mask_2 = self.out_1(decoder_output), self.out_2(decoder_output)
            return mask_1, mask_2, torch.sigmoid(x)
        else:
            decoder_output = self.Unet.decoder(*features)
            mask_1, mask_2 = self.out_1(decoder_output), self.out_2(decoder_output)      
            return mask_1, mask_2

# TODO Custome model 