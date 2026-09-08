import numpy as np

from backbones.res_u_net import ResUNet
from backbones.easy_res_u_net import EasyResUNet
from backbones.inverted_residual import *
import torch
from backbones.fasmodel import FASModel


class ATR_FAS(nn.Module):
    def __init__(self, infer_type=None, frame_num=6, **kwargs):
        super(ATR_FAS, self).__init__()
        self.infer_type = infer_type
        self.frame_num = frame_num
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # self.gate = nn.Sequential(
        #     DownConvNormAct(3, 32),
        #     DownConvNormAct(32, 64),
        #     DownConvNormAct(64, 64, kernel_size=7),  # [32, 32, 64]
        #     torch.nn.AdaptiveAvgPool2d((1, 1)),  # [1, 1, 64]
        #     Reshape(64),
        #     nn.Linear(64, 32),
        #     nn.Linear(32, 3),
        # )
        self.gate = FASModel(3, backbone='resnet34')

        # checkpoint = torch.load('./backbones/epoch_72_best_0.0.pkl')
        # model_state = self.gate.state_dict()
        # pretrained_state = checkpoint['network']
        # pretrained_state = {k: v for k, v in pretrained_state.items() if
        #                     k in model_state and v.size() == model_state[k].size()}
        # model_state.update(pretrained_state)
        # self.gate.load_state_dict(model_state)
        #
        # # Freeze all parameters in model.features
        # for param in self.gate.parameters():
        #     param.requires_grad = False

        # [3, 256, 256]->[64, 64, 64]
        self.head_stem = nn.Sequential(
            DownConvNormAct(3, 32),
            DownConvNormAct(32, 64),
        )

        self.pos_embedding = torch.nn.Parameter(torch.randn(1, 64, 64, 64))
        torch.save(self.pos_embedding, 'pos_embedding.pt')
        loaded_pos_embedding = torch.load('pos_embedding.pt')

        # 注意力值 Attention value
        # 1、用来融合（预测出的）多帧深度图成一帧，然后再分类 Used to fuse (predicted) multiple frames of depth maps into one frame and then classify
        self.attention_stem = nn.Sequential(
            DownConvNormAct(3, 16),
            DownConvNormAct(16, 32),
        )
        self.easy_u_net_attention = EasyResUNet()

        # 三个任务 Three tasks
        self.u_net_real = ResUNet()
        self.u_net_pad = ResUNet()
        self.u_net_df = ResUNet()

        depth_map_cor = np.reshape(np.arange(256) / 255., [1, 1, 1, -1]).astype(np.float32)
        self.depth_map_cof = torch.from_numpy(depth_map_cor)

        # 输出分类 Output classification
        self.f_net = nn.Sequential(
            DownConvNormAct(1, 16),  # 32*32*16
            ConvNormAct(16, 8, 3),  # 32*32*8
            ConvNormAct(8, 4, 3),  # 32*32*4
            Reshape(32 * 32 * 4),
            L2Normalize(1),
            nn.Linear(32 * 32 * 4, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        self.softmax = nn.Softmax(dim=-1)


    def to(self, device):
        self.pos_embedding.to(device)
        self.depth_map_cof = self.depth_map_cof.to(device)

        self.gate.to(device)
        self.head_stem.to(device)
        self.attention_stem.to(device)
        self.easy_u_net_attention.to(device)
        self.u_net_real.to(device)
        self.u_net_pad.to(device)
        self.u_net_df.to(device)
        self.f_net.to(device)
        self.softmax.to(device)

        return super(ATR_FAS, self).to(device)

    # def forward(self, x1, x2):
    #
    #     xs = torch.split(x1, split_size_or_sections=256, dim=3)
    #
    #     if self.training:
    #         loc, conf, domain_invariant, gate, gate_feat1, gate_feat2 = self.gate(x1, x2)
    #     else:
    #         gate = self.gate(x1, x2)
    #     # gate = self.gate(x1)
    #
    #     gate_soft_max = torch.softmax(gate, dim=1)
    #     atten_x = self.attention_stem(xs[1])
    #     depth_map_attention = self.infer_depth_map_attention(atten_x)
    #
    #     x = self.head_stem(xs[1]) + self.pos_embedding
    #
    #     # 做深度图 Make a depth map
    #     single_depth_map, depth_soft_max, depth_map = (
    #         self.infer_depth_map([self.u_net_real, self.u_net_pad, self.u_net_df],
    #                              x, depth_map_attention, gate_soft_max))
    #
    #     # 做分类 Do classification
    #     final_cls = self.f_net(single_depth_map)
    #
    #     if self.training:
    #         return loc, conf, domain_invariant, gate, gate_feat1, gate_feat2, single_depth_map, final_cls
    #         # return single_depth_map, final_cls
    #     else:
    #         return self.softmax(final_cls)

    def forward(self, x1):

        xs = torch.split(x1, split_size_or_sections=256, dim=3)

        gate = self.gate(x1, x1)

        gate_soft_max = torch.softmax(gate, dim=1)
        atten_x = self.attention_stem(xs[1])
        depth_map_attention = self.infer_depth_map_attention(atten_x)

        x = self.head_stem(xs[1]) + self.pos_embedding

        # 做深度图 Make a depth map
        single_depth_map, depth_soft_max, depth_map = (
            self.infer_depth_map([self.u_net_real, self.u_net_pad, self.u_net_df],
                                 x, depth_map_attention, gate_soft_max))

        # 做分类 Do classification
        final_cls = self.f_net(single_depth_map)

        return self.softmax(final_cls), single_depth_map, gate_soft_max


    def infer_depth_map_attention(self, x):
        attention_x = self.easy_u_net_attention(x)
        attention_soft_max = self.pixel_wise_softmax(attention_x)
        attention_map = torch.unsqueeze(torch.sum(self.depth_map_cof.to(attention_soft_max.device) * attention_soft_max, dim=-1), dim=1)
        attention_map = torch.softmax(torch.reshape(attention_map, [-1, self.frame_num, 64, 64]), dim=1)
        return attention_map

    def infer_depth_map(self, u_net, x, depth_map_attention, gate_soft_max):
        gate_soft_max = torch.reshape(gate_soft_max, [-1, 3, 1, 1])
        depth_x = u_net[0](x) * gate_soft_max[:, 0:1, :, :] + u_net[1](x) * gate_soft_max[:, 1:2, :, :] + u_net[2](x) * gate_soft_max[:, 2:3, :, :]
        depth_soft_max = self.pixel_wise_softmax(depth_x)
        depth_map = torch.unsqueeze(torch.sum(self.depth_map_cof.to(depth_soft_max.device) * depth_soft_max, dim=-1), dim=1)

        # 融合多帧深度图成单帧 Fusion of multiple depth maps into a single frame
        single_depth_map = torch.sum(
            torch.reshape(depth_map, [-1, self.frame_num, 64, 64]) * depth_map_attention, dim=1,
            keepdim=True)
        return single_depth_map, depth_soft_max, depth_map

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                print("copy value to %s" % name)
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

    @classmethod
    def pixel_wise_softmax(cls, x):
        # 将像素交换到最后的维度 Swap pixels to the last dimension
        x = x.permute(0, 2, 3, 1)
        channel_max, _ = torch.max(x, dim=3, keepdim=True)
        exponential_map = torch.exp(x - channel_max)
        normalize = torch.sum(exponential_map, dim=3, keepdims=True)
        return exponential_map / (normalize + 1e-5)


if __name__ == "__main__":
    model = ATR_FAS(frame_num=1)

    x = torch.randn(1, 3, 256, 512)

    model.train()
    ret = model(x, x)

    model.eval()
    dummy_input = torch.randn(1, 3, 256, 512)
    torch.onnx.export(model, dummy_input, "out.onnx", keep_initializers_as_inputs=False, verbose=False,
                      opset_version=12)

    import onnx

    onnx_model = onnx.load("out.onnx")
    from onnxsim import simplify
    onnx_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    import onnxoptimizer
    onnx_model = onnxoptimizer.optimize(onnx_model)
    onnx.save(onnx_model, "out.onnx")

    from onnx import numpy_helper

    total_parameters = 0
    for initializer in onnx_model.graph.initializer:
        total_parameters += numpy_helper.to_array(initializer).size
    final_cls = model(x)
    final_cls = final_cls
