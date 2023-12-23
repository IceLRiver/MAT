import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class PatchesT(nn.Module):
    def __init__(self, patch_sizeT=100):
        super(PatchesT, self).__init__()
        self.patch_size = patch_sizeT

    def forward(self, images):
        batch_size, channels, height, width = images.size()
        patches = images.view(batch_size, channels, height*width // self.patch_size, self.patch_size)
        return patches

class PatchEncoderT(nn.Module):
    def __init__(self, num_patchesT, patch_sizeT, projection_dimT=64):
        super(PatchEncoderT, self).__init__()
        self.num_patches = num_patchesT
        self.projection_dimT = projection_dimT
        self.projection = nn.Linear(patch_sizeT, self.projection_dimT)
        self.position_embedding = nn.Embedding(
            num_embeddings=self.num_patches, embedding_dim=self.projection_dimT
        )

    def forward(self, patch):
        positions = torch.arange(0, self.num_patches).to(patch.device)
        positions = positions.unsqueeze(0).unsqueeze(0)
        positions = positions.repeat(patch.size(0), patch.size(1), 1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class PatchEncoder(nn.Module):
    def __init__(self, input_dim, num_patches, max_fre, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.type_patches = max_fre
        self.projection2 = nn.Linear(input_dim, projection_dim).to('cuda')
        self.position_embedding = nn.Embedding(
            num_embeddings=self.type_patches, embedding_dim=projection_dim
        ).to('cuda')

    def forward(self, patch):
        batch_size = patch.size(0)
        positions = patch[:, :, -1]
        positions = positions.view(batch_size, self.num_patches).to(torch.int)
        patch = patch[:, :, :-1]
        encoded = self.projection2(patch) + self.position_embedding(positions)
        return encoded

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rate):
        super(MLP, self).__init__()
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.dense1 = nn.Linear(input_dim, hidden_units[0])
        self.dense2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        x = F.gelu(self.dense1(x))
        x = self.dropout(x)
        x = F.gelu(self.dense2(x))
        x = self.dropout(x)
        return x


# Mode编码器
class ModeEncoder(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(ModeEncoder, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, batch_first=True).to('cuda'),
            num_layers=num_layers
        )

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = self.transformer(x.view(batch_size * channels, height, width))
        return x.view(batch_size, channels, height, width)

# Transformer编码器
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(TransformerEncoder, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, batch_first=True).to('cuda'),
            num_layers=num_layers
        )

    def forward(self, x):
        x = self.transformer(x)
        return x


class MAT(nn.Module):
    def __init__(self, input_shape, num_classes, max_fre, patch_size=1201, patch_sizeT=100, projection_dim=64,
                 num_heads=4, transformer_layers=4):
        super(MAT, self).__init__()
        self.input_shape_all = input_shape
        self.num_classes = num_classes
        self.max_fre = max_fre
        self.patch_size = patch_size
        self.patch_sizeT = patch_sizeT
        self.num_patches = input_shape[0] * input_shape[1] // patch_size
        self.num_patchesT = ((patch_size - 1) // patch_sizeT)
        self.projection_dimT = projection_dim
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.mlp_head_units = [2048, 512]
        self.transformer_units = [projection_dim * 2, projection_dim]

        self.ModeEncoder = nn.Sequential()
        self.ModeEncoder.add_module('PatchesT', PatchesT(patch_sizeT=patch_sizeT))
        self.ModeEncoder.add_module('PatchEncoderT', PatchEncoderT(self.num_patchesT, self.patch_sizeT, self.projection_dimT))
        self.ModeEncoder.add_module('ModeEncoderT', ModeEncoder(self.projection_dimT, self.transformer_layers))
        self.ModeEncoder.add_module('Norm', nn.LayerNorm(normalized_shape=self.projection_dimT))

        self.mlp1 = MLP(input_dim=self.num_patchesT*self.projection_dimT, hidden_units=[unit // 2 for unit in self.mlp_head_units], dropout_rate=0.3)

        self.PatchEncoder = nn.Sequential()
        self.PatchEncoder.add_module('PatchEncoder', PatchEncoder(self.mlp_head_units[1]//2, self.num_patches, self.max_fre, self.projection_dim))
        self.PatchEncoder.add_module('TransEncoder', TransformerEncoder(self.projection_dim, self.transformer_layers))
        self.PatchEncoder.add_module('Norm', nn.LayerNorm(normalized_shape=self.projection_dim))
        
        self.mlp2 = MLP(input_dim=self.num_patches*self.projection_dim, hidden_units=self.mlp_head_units, dropout_rate=0.4)

        self.output_layer = nn.Linear(self.mlp_head_units[1], self.num_classes)

    def forward(self, x):
        patchesT = x[:, :, :-1, :]
        positionf = x[:, :, -1, :]

        encoded_patchesT = self.ModeEncoder(patchesT)

        representation1 = encoded_patchesT.view(encoded_patchesT.size(0), encoded_patchesT.size(1), -1)
        representation1 = F.dropout(representation1, p=0.3)
        featuresT = self.mlp1(representation1)
        patches = torch.cat([featuresT, positionf], dim=2)

        encoded_patches = self.PatchEncoder(patches)

        representation2 = encoded_patches.view(encoded_patches.size(0), -1)
        representation2 = F.dropout(representation2, p=0.3)
        features = self.mlp2(representation2)

        logits = self.output_layer(features)
        return logits

if __name__ == "__main__":
    input_shape = (11, 1201, 1)
    num_classes = 26
    max_fre = 12000
    mat = MAT(input_shape, num_classes, max_fre)
    # print(mat)
    summary(mat)