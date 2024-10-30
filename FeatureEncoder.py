class DiscriminativeFeatureAttentionNetwork(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(DiscriminativeFeatureAttentionNetwork, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // reduction_ratio)
        self.fc2 = nn.Linear(input_channels // reduction_ratio, input_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pooled = self.global_avg_pool(x).view(b, c)
        fc1_out = self.relu(self.fc1(avg_pooled))
        fc2_out = self.sigmoid(self.fc2(fc1_out))
        attention_weights = fc2_out.view(b, c, 1, 1)
        out = x * attention_weights  
        return out

# ResNet101 with integrated Discriminative Feature Attention for multi-class classification
class ResNet34ForMultiClassWithAttention(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(ResNet34ForMultiClassWithAttention, self).__init__()
        self.base_model = models.resnet34(pretrained=pretrained)
        self.base_model.fc = nn.Linear(
            in_features=self.base_model.fc.in_features,
            out_features=num_classes
        )
        
        self.discriminative_attention = DiscriminativeFeatureAttentionNetwork(input_channels=256)  

    def forward(self, image, targets=None):
        batch_size = image.size(0)
        x = self.base_model.conv1(image)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.discriminative_attention(x)
        x = self.base_model.layer4(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.base_model.fc(x)
        if targets is not None:
            loss = nn.CrossEntropyLoss()(out, targets)
            return out, loss
        
        return out
