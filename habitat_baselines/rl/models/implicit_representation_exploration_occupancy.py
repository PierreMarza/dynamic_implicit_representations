import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MLP_expl_occ(nn.Module):
    def __init__(self, D=3, W=512, input_ch=2, output_ch=2):
        super().__init__()
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) for i in range(D - 2)]
        )
        self.out = nn.Linear(W, output_ch)

    def forward(self, h):
        for i in range(len(self.pts_linears)):
            h = self.pts_linears[i](h)
            h = F.relu(h)
        return self.out(h)

    def reset_parameters(self):
        for i in range(len(self.pts_linears)):
            self.pts_linears[i].reset_parameters()
        self.out.reset_parameters()


def run_epoch_expl_occ(
    criterion, net_expl_occ, optimizer_expl_occ, xyz, labels, to_train
):
    xyz_pred = []
    for i in range(xyz.shape[0]):
        if to_train[i]:
            xyz_pred_ = net_expl_occ[i](xyz[i])
        else:
            xyz_pred_ = None
        xyz_pred.append(xyz_pred_)

    losses = []
    for i in range(xyz.shape[0]):
        if xyz_pred[i] is not None:
            loss = criterion(xyz_pred[i].view(-1, 3), labels[i].view(-1))
            optimizer_expl_occ[i].zero_grad()
            loss.backward()
            optimizer_expl_occ[i].step()
            losses.append(loss.item())
        else:
            losses.append(-1)
    return losses


class PositionalEncoding(nn.Module):
    # Inspired from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout=0.1, nb_mlp_neurons=64 + 3):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(nb_mlp_neurons + 1).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(nb_mlp_neurons + 1, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x += self.pe[:, 0, :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    # From https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(
        self,
        d_in_w_1,
        d_in_w_2,
        d_in_w_3,
        d_model,
        nhead,
        d_hid,
        nlayers,
        dropout,
        device,
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(
            d_model, dropout, nb_mlp_neurons=512 + 512 + 3
        )
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

        self.encoder_w_1 = nn.Sequential(
            nn.Linear(d_in_w_1, d_model),
        )

        self.encoder_w_2 = nn.Sequential(
            nn.Linear(d_in_w_2, d_model),
        )

        self.encoder_w_3 = nn.Sequential(
            nn.Linear(d_in_w_3, d_model),
        )

        self.cls_embedding = nn.Embedding(1, d_model)
        self.device = device

    def forward(self, w_1, w_2, w_3):
        w_1 = self.encoder_w_1(w_1)
        w_2 = self.encoder_w_2(w_2)
        w_3 = self.encoder_w_3(w_3)

        cls_token = self.cls_embedding(
            torch.zeros(w_1.shape[0], 1).type(torch.LongTensor).to(self.device)
        )
        src = torch.cat([cls_token, w_1, w_2, w_3], dim=1)

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)

        return output[:, 0, :]  # only representation of cls token


class MLPReader(nn.Module):
    def __init__(
        self,
        device,
        d_in_w_1=41,
        d_in_w_2=513,
        d_in_w_3=513,
        d_model=3 * 3 * 64,
        nhead=8,
        d_hid=3 * 3 * 64,
        nlayers=4,
        dropout=0.1,
        d_gps_input=2,
        d_gps_output=32,
        heading_num_embeddings=12,
        d_heading=32,
    ):
        super().__init__()
        self.encoder = TransformerModel(
            d_in_w_1=d_in_w_1,
            d_in_w_2=d_in_w_2,
            d_in_w_3=d_in_w_3,
            d_model=d_model,
            nhead=nhead,
            d_hid=d_hid,
            nlayers=nlayers,
            dropout=dropout,
            device=device,
        )

        self.fc_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
        )

        self.fc_gps = nn.Sequential(nn.Linear(d_gps_input, d_gps_output), nn.ReLU(True))
        self.embed_curr_heading = nn.Embedding(
            num_embeddings=heading_num_embeddings, embedding_dim=d_heading
        )

        self.fc_encoder_shift = nn.Sequential(
            nn.Linear(d_model + d_gps_output, d_model * 2), nn.ReLU(True)
        )
        self.fc_encoder2_shift = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(True)
        )

        self.fc_encoder_rot = nn.Sequential(
            nn.Linear(d_model + d_heading, d_model * 2), nn.ReLU(True)
        )
        self.fc_encoder2_rot = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(True)
        )

    def forward(self, w_1, w_2, w_3, gps, curr_heading):
        x = self.encoder(w_1, w_2, w_3)
        x = self.fc_encoder(x)

        gps = self.fc_gps(gps)
        curr_heading = self.embed_curr_heading(curr_heading)

        x = torch.cat((x, gps), dim=-1)
        x_shift = self.fc_encoder_shift(x)
        x_shift = self.fc_encoder2_shift(x_shift)

        x = torch.cat((x_shift, curr_heading[:, 0, :]), dim=-1)
        x = self.fc_encoder_rot(x)
        x = self.fc_encoder2_rot(x)

        return x


class DecoderModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 8, 3, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2, padding=0, output_padding=1),
        )

    def forward(self, x):
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x


def reading_net_expl_occ_weights(net_list_exploration, device):
    w_1_batch = []
    w_2_batch = []
    w_3_batch = []
    for net in net_list_exploration:
        state_dict = net.state_dict()

        w_1_ = state_dict["pts_linears.0.weight"].detach()
        w_1_b = state_dict["pts_linears.0.bias"].detach()
        w_1 = torch.cat([w_1_, w_1_b.unsqueeze(-1)], dim=1).unsqueeze(0).to(device)
        w_1_batch.append(w_1)

        w_2_ = state_dict["pts_linears.1.weight"].detach()
        w_2_b = state_dict["pts_linears.1.bias"].detach()
        w_2 = torch.cat([w_2_, w_2_b.unsqueeze(-1)], dim=1).unsqueeze(0).to(device)
        w_2_batch.append(w_2)

        w_3_ = state_dict["out.weight"].detach()
        w_3_b = state_dict["out.bias"].detach()
        w_3 = torch.cat([w_3_, w_3_b.unsqueeze(-1)], dim=1).unsqueeze(0).to(device)
        w_3_batch.append(w_3)

    w_1_batch = torch.cat(w_1_batch, dim=0)
    w_2_batch = torch.cat(w_2_batch, dim=0)
    w_3_batch = torch.cat(w_3_batch, dim=0)

    return w_1_batch, w_2_batch, w_3_batch
