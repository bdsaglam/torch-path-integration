import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_path_integration.utils import init_trunc_normal


class PathIntegrationModule(nn.Module):
    def __init__(self,
                 dim_vel,
                 n_place_cells,
                 n_hd_cells,
                 lstm_hidden_size=128,
                 grid_layer_size=256,
                 grid_layer_dropout_rate=0.5,
                 ):
        super().__init__()
        self.dim_vel = dim_vel
        self.grid_layer_size = grid_layer_size
        self.n_place_cells = n_place_cells
        self.n_hd_cells = n_hd_cells

        self.lstm_cell = nn.LSTMCell(dim_vel, hidden_size=lstm_hidden_size)
        self.grid_layer = nn.Linear(lstm_hidden_size, grid_layer_size, bias=False)
        self.dropout = nn.Dropout(grid_layer_dropout_rate)
        self.place_layer = nn.Linear(grid_layer_size, n_place_cells)
        self.head_direction_layer = nn.Linear(grid_layer_size, n_hd_cells)

        self.initial_hx = nn.Linear(n_place_cells + n_hd_cells, lstm_hidden_size)
        self.initial_cx = nn.Linear(n_place_cells + n_hd_cells, lstm_hidden_size)

        with torch.no_grad():
            init_trunc_normal(self.initial_hx.weight, lstm_hidden_size)
            init_trunc_normal(self.initial_cx.weight, lstm_hidden_size)
            init_trunc_normal(self.grid_layer.weight, grid_layer_size)
            init_trunc_normal(self.place_layer.weight, n_place_cells)
            init_trunc_normal(self.head_direction_layer.weight, n_hd_cells)

            nn.init.zeros_(self.initial_hx.bias)
            nn.init.zeros_(self.initial_cx.bias)
            nn.init.zeros_(self.place_layer.bias)
            nn.init.zeros_(self.head_direction_layer.bias)

    def initial_hidden_state(self, place, head_direction):
        x = torch.cat([place, head_direction], -1)
        hx = self.initial_hx(x)
        cx = self.initial_cx(x)
        return hx, cx

    def forward(self, velocity, hidden_state):
        assert len(velocity.shape) == 2

        hx, cx = self.lstm_cell(velocity, hidden_state)
        grid_activations = self.dropout(self.grid_layer(hx))
        place = F.softmax(self.place_layer(grid_activations), -1)
        head_direction = F.softmax(self.head_direction_layer(grid_activations), -1)

        return grid_activations, place, head_direction, (hx, cx)

    def reg_loss(self):
        return self.grid_layer.weight.norm(2)
