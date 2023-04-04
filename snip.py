class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc1 = nn.Conv2d(d_in, 16, (1, 1), bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(16, d_in, (1, 1), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.mlp = nn.Conv2d(d_in, d_out, kernel_size=(1,1))
        self.bn = nn.BatchNorm2d(d_out)

    def forward(self, feature_set):

        att_activation = self.fc1(feature_set)
        att_activation = self.relu(att_activation)
        att_activation = self.fc2(att_activation)
        att_scores = self.sigmoid(att_activation)

        f_agg = feature_set * att_scores
        f_agg = torch.mean(f_agg, dim=3, keepdim=True)
        f_agg = self.bn(self.mlp(f_agg))
        f_agg = f_agg.squeeze()
        return f_agg
