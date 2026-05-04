# Ref:
# 1. https://github.com/li-Pingan/fourier-neural-operator
# 2. https://github.com/neuraloperator/neuraloperator

from typing import List, Tuple, Union
Number = Union[float, int]

import torch
from torch import nn
import torch.nn.functional as F



def regular_grid_nd(resolutions: List[int], grid_boundaries: List[List[int]]=[[0,1]] * 2):
    """regular_grid_nd generates a tensor of coordinate points that 
    describe a bounded regular grid.
    
    Creates a dim x res_d1 x ... x res_dn stack of positional encodings A, where
    A[:,c1,c2,...] = [[d1,d2,...dn]] at coordinate (c1,c2,...cn) on a (res_d1, ...res_dn) grid. 

    Parameters
    ----------
    resolutions : List[int]
        resolution of the output grid along each dimension
    grid_boundaries : List[List[int]], optional
        List of pairs [start, end] of the boundaries of the
        regular grid. Must correspond 1-to-1 with resolutions default [[0,1], [0,1]]

    Returns
    -------
    grid: tuple(Tensor)
    list of tensors describing positional encoding 
    """
    assert len(resolutions) == len(grid_boundaries), "Error: inputs must have same number of dimensions"
    dim = len(resolutions)

    meshgrid_inputs = list()
    for res, (start,stop) in zip(resolutions, grid_boundaries):
        meshgrid_inputs.append(torch.linspace(start, stop, res + 1)[:-1])
    grid = torch.meshgrid(*meshgrid_inputs, indexing='ij')
    grid = tuple([x.repeat([1]*dim) for x in grid])
    return grid


class GridEmbeddingND(nn.Module):
    """A positional embedding as a regular ND grid
    """
    def __init__(self, in_channels: int, dim: int=2, grid_boundaries=[[0, 1], [0, 1]]):
        """GridEmbeddingND applies a simple positional 
        embedding as a regular ND grid

        Parameters
        ----------
        in_channels : int
            number of channels in input
        dim : int
            dimensions of positional encoding to apply
        grid_boundaries : list, optional
            coordinate boundaries of input grid along each dim, by default [[0, 1], [0, 1]]
        """
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        assert self.dim == len(grid_boundaries), f"Error: expected grid_boundaries to be\
            an iterable of length {self.dim}, received {grid_boundaries}"
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None
    
    @property
    def out_channels(self):
        return self.in_channels + self.dim

    def grid(self, spatial_dims: torch.Size, device: str, dtype: torch.dtype):
        """grid generates ND grid needed for pos encoding
        and caches the grid associated with MRU resolution

        Parameters
        ----------
        spatial_dims : torch.Size
             sizes of spatial resolution
        device : literal 'cpu' or 'cuda:*'
            where to load data
        dtype : str
            dtype to encode data

        Returns
        -------
        torch.tensor
            output grids to concatenate 
        """
        # handle case of multiple train resolutions
        if self._grid is None or self._res != spatial_dims: 
            grids_by_dim = regular_grid_nd(spatial_dims,
                                      grid_boundaries=self.grid_boundaries)
            # add batch, channel dims
            grids_by_dim = [x.to(device).to(dtype).unsqueeze(0).unsqueeze(0) for x in grids_by_dim]
            self._grid = grids_by_dim
            self._res = spatial_dims

        return self._grid
    
    def forward(self, data, batched=True):
        """
        Params
        --------
        data: torch.Tensor
            assumes shape batch (optional), channels, x_1, x_2, ...x_n
        batched: bool
            whether data has a batch dim
        """
        # add batch dim if it doesn't exist
        if not batched:
            if data.ndim == self.dim + 1:
                data = data.unsqueeze(0)
        batch_size = data.shape[0]
        grids = self.grid(spatial_dims=data.shape[2:],
                          device=data.device,
                          dtype=data.dtype)
        grids = [x.repeat(batch_size, *[1] * (self.dim+1)) for x in grids]
        out =  torch.cat((data, *grids),
                         dim=1)
        return out

class Flattened1dConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, bias=False):
        """Flattened3dConv is a Conv-based skip layer for
        input tensors of ndim > 3 (batch, channels, d1, ...) that flattens all dimensions 
        past the batch and channel dims into one dimension, applies the Conv,
        and un-flattens.

        Parameters
        ----------
        in_channels : int
            in_channels of Conv1d
        out_channels : int
            out_channels of Conv1d
        kernel_size : int
            kernel_size of Conv1d
        bias : bool, optional
            bias of Conv3d, by default False
        """
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              bias=bias)
    def forward(self, x):
        # x.shape: b, c, x1, ..., xn x_ndim > 1
        size = list(x.shape)
        # flatten everything past 1st data dim
        x = x.view(*size[:2], -1)
        x = self.conv(x)
        # reshape x into an Nd tensor b, c, x1, x2, ...
        x = x.view(size[0], self.conv.out_channels, *size[2:])
        return x
    

class InstanceNorm(nn.Module):
    def __init__(self, **kwargs):
        """InstanceNorm applies dim-agnostic instance normalization
        to data as an nn.Module. 

        kwargs: additional parameters to pass to instance_norm() for use as a module
        e.g. eps, affine
        """
        super().__init__()
        self.kwargs = kwargs
    
    def forward(self, x):
        size = x.shape
        x = torch.nn.functional.instance_norm(x, **self.kwargs)
        assert x.shape == size
        return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes[0] #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes[1]

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    
    
class FNOBlocks(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        n_layers=1,
        non_linearity: nn.Module=F.gelu,
    ):
        super().__init__()
        self.non_linearity = non_linearity
        self.n_layers = n_layers
        
        self.convs = nn.ModuleList([
                SpectralConv2d(
                in_channels,
                out_channels,
                n_modes,
            ) 
            for _ in range(n_layers)])

        self.fno_skips = nn.ModuleList(
            [
                Flattened1dConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False,)
                for _ in range(n_layers)
            ]
        )

        # Each block will have 2 norms if we also use a ChannelMLP
        self.norm = nn.ModuleList(
                [
                    InstanceNorm()
                    for _ in range(n_layers)
                ]
            )

    def forward(self, x, index=0):
        x_skip_fno = self.fno_skips[index](x)
        x_fno = self.convs[index](x)

        if self.norm is not None:
            x_fno = self.norm[index](x_fno)

        x = x_fno + x_skip_fno

        if (index < (self.n_layers - 1)):
            x = self.non_linearity(x)

        return x
    
    
class ChannelMLP(nn.Module):
    """ChannelMLP applies an arbitrary number of layers of 
    1d convolution and nonlinearity to the channels of input
    and is invariant to spatial resolution.

    Parameters
    ----------
    in_channels : int
    out_channels : int, default is None
        if None, same is in_channels
    hidden_channels : int, default is None
        if None, same is in_channels
    n_layers : int, default is 2
        number of linear layers in the MLP
    non_linearity : default is F.gelu
    dropout : float, default is 0
        if > 0, dropout probability
    last_bias : bool, default is True
        if True, adds a bias term to the last layer
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        hidden_channels=None,
        n_layers=2,
        n_dim=2,
        non_linearity=F.gelu,
        dropout=0.0,
        bias=True,
        last_bias=True,
        **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (
            in_channels if hidden_channels is None else hidden_channels
        )
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
            if dropout > 0.0
            else None
        )
        
        # we use nn.Conv1d for everything and roll data along the 1st data dim
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0 and i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.in_channels, self.out_channels, 1, bias=bias and (last_bias if i == (n_layers - 1) else True)))
            elif i == 0:
                self.fcs.append(nn.Conv1d(self.in_channels, self.hidden_channels, 1, bias=bias and (last_bias if i == (n_layers - 1) else True)))
            elif i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.out_channels, 1, bias=bias and (last_bias if i == (n_layers - 1) else True)))
            else:
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.hidden_channels, 1, bias=bias and (last_bias if i == (n_layers - 1) else True)))

    def forward(self, x):
        reshaped = False
        size = list(x.shape)
        if x.ndim > 3:  
            # batch, channels, x1, x2... extra dims
            # .reshape() is preferable but .view()
            # cannot be called on non-contiguous tensors
            x = x.reshape((*size[:2], -1)) 
            reshaped = True

        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        # if x was an N-d tensor reshaped into 1d, undo the reshaping
        # same logic as above: .reshape() handles contiguous tensors as well
        if reshaped:
            x = x.reshape((size[0], self.out_channels, *size[2:]))

        return x
    
    

class FNO(nn.Module):
    """Fourier Neural Operator
    """
    def __init__(
        self,
        n_modes: Tuple[int],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int=4,
        non_linearity: nn.Module=F.gelu,
        bias=True,
    ):
        
        super().__init__()
        self.n_layers = n_layers
        
        spatial_grid_boundaries = [[0., 1.]] * len(n_modes)
        self.positional_embedding = GridEmbeddingND(
            in_channels=in_channels,
            dim=len(n_modes), 
            grid_boundaries=spatial_grid_boundaries
        )

        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=n_modes,
            n_layers=n_layers,
            non_linearity=non_linearity,
        )
        
        lifting_in_channels = in_channels + (len(n_modes) if self.positional_embedding is not None else 0)
        self.lifting = ChannelMLP(
            in_channels=lifting_in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            n_layers=1,
            n_dim=len(n_modes),
            non_linearity=non_linearity,
            bias=bias,
        )

        self.projection = ChannelMLP(
            in_channels=hidden_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=2,
            n_dim=len(n_modes),
            non_linearity=non_linearity,
            bias=bias,
            last_bias=False
        )

    def forward(self, x):
        # append spatial pos embedding if set
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)
        
        x = self.lifting(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx)

        x = self.projection(x)

        return x