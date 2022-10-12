import numpy as np
import time
import torch
from torch import tensor
from torch.cuda.amp import autocast
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm


class small_conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch, do_batch_norm=False):
        super(small_conv_block, self).__init__()
        self.do_batch_norm = do_batch_norm
        if self.do_batch_norm is False:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1), dilation=(1,1,1), bias=True),
                nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, do_batch_norm=False):
        super(up_conv, self).__init__()
        self.do_batch_norm = do_batch_norm
        if self.do_batch_norm is False:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=(1, 2, 2)),
                nn.Conv3d(in_ch, out_ch, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1), dilation=(1,1,1), bias=True),
                nn.ReLU(inplace=True))
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(negative_slope=0.2, inplace=True) )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    """
    Attention Block
    """
    def __init__(self, F_g, F_l, F_int, do_batch_norm=False):
        super(Attention_block, self).__init__()
        self.do_batch_norm = do_batch_norm
        if self.do_batch_norm is False:
            self.W_g = nn.Sequential(
                nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            )

            self.W_x = nn.Sequential(
                nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            )

            self.psi = nn.Sequential(
                nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Tanh()
            )
        else:
            self.W_g = nn.Sequential(
                nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(F_int)
            )

            self.W_x = nn.Sequential(
                nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(F_int)
            )

            self.psi = nn.Sequential(
                nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(1),
                nn.Tanh()
            )

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class UNet3DLite(nn.Module):
    """
    UNet 3D small model
    Expects 4 RGB images at input and outputs 3 interpolated RGB images
    """
    def __init__(self, n_input_channels=3, n_output_channels=3, init_filters=16):
        super(UNet3DLite, self).__init__()

        n1 = init_filters
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]

        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))

        self.Conv1 = small_conv_block(n_input_channels, filters[0])
        self.Conv2 = small_conv_block(filters[0], filters[1])
        self.Conv3 = small_conv_block(filters[1], filters[2])
        self.Conv4 = small_conv_block(filters[2], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = small_conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = small_conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=int(filters[0]/2))
        self.Up_conv2 = small_conv_block(filters[1], filters[0])

        self.Conv = nn.Conv3d(filters[0], n_output_channels, kernel_size=(4, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.network_summary()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.pool(e1)
        e2 = self.Conv2(e2)

        e3 = self.pool(e2)
        e3 = self.Conv3(e3)

        e4 = self.pool(e3)
        e4 = self.Conv4(e4)

        d4 = self.Up4(e4)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def network_summary(self):
        for layer in self.children():
            print(layer)
            for name, param in layer.named_parameters():
                print('\t%s %s' % (name, param.shape))

        total_parameters = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            total_parameters += nn
        print("Total parameters: " + str(total_parameters))
        print("Total trainable parameters: " + str(self.count_parameters()))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = 'dml'
    model = UNet3DLite(init_filters=16)
    model.to(device)
    model.eval()

    dummy_input = torch.rand((1, 3, 4, 480, 640)).to(device)
    model = torch.jit.trace(model, dummy_input)

    #torch.backends.cudnn.benchmark = True
    for batch_size in range(1, 4):
        dummy_input = torch.rand((batch_size, 3, 4, 480, 640)).to(device)

        repetitions = 10000
        timings=np.zeros((repetitions,1))
        print('Running inference on {}'.format(device))

        if 'cpu' == device:
            #CPU-WARM-UP
            for _ in range(10):
                _ = model(dummy_input)
                    # MEASURE PERFORMANCE
            with torch.no_grad():
                for rep, _ in zip(range(repetitions), tqdm(range(repetitions))):
                    start = time.time()
                    _ = model(dummy_input)
                    end = time.time()
                    # WAIT FOR GPU SYNC
                    torch.cuda.synchronize()
                    curr_time = end-start
                    timings[rep] = curr_time
            mean_syn = (np.sum(timings) / repetitions)
            print('GPU mean sample time {.2d} ms'.format(mean_syn))
            print('GPU total sample time {.2d} ms'.format(np.sum(timings)))
        else:
            # INIT LOGGERS
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            #GPU-WARM-UP
            for _ in range(10):
                _ = model(dummy_input)
            # MEASURE PERFORMANCE
            with torch.no_grad():
                for rep, _ in zip(range(repetitions), tqdm(range(repetitions))):
                    starter.record()
                    with autocast(enabled=True):
                        out = model(dummy_input)
                    ender.record()
                    # WAIT FOR GPU SYNC
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    timings[rep] = curr_time
            mean_syn = (np.sum(timings) / repetitions)

            print('GPU mean time {} ms for per sample'.format(mean_syn/batch_size))
            print('GPU total time {} ms'.format(np.sum(timings)/batch_size))
