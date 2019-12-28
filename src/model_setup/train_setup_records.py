import numpy as np

from model_setup.train_setup_ddff_mod import \
    TrainSetupDDFFBlender, TrainSetupDDFFRNN, TrainSetupDenseNet3DAutoEncoder, TrainSetupCoCDepthNet, \
    TrainSetupDepth, TrainSetupCoCFgbgNet, TrainSetupSignedCoC, TrainSetupCoC

from net.pool_net import PoolNet, PoolNetEncoder, PoolNetDecoder
from net.coc_depth_net import CoCDepthEncShareNet, CoCDepthLayeredNet, CoCFgbgLayeredPoolNet, FgbgCocMultiDecoderPoolNet
from net.unet import UNetEncoder, UNetDecoder, UNet
from net.net_stack import ConcatNetStackSandwitch
from net.ddff_net import DDFFNet
from net.dummy_nets import DDFFNetDummy, DDFFNetCoCDummy, DDFFNetFgbgCoCDummy
#from net.recurrent_ae_web import RecurrentAEWeb
from net.modules import ParallelNet, MaxReduce, InputSplitModule, InputSubsetModule, AE2Decoder, OutputSplitModule
from net.consec_net import ConsecNet, PreDepthConsecNet
from net.ae_rnn import AERNN
from net.sliding_window_net import SlidingWindowNet
from net.vgg import VGGAE
from net.recurrent_ae import RecurrentAE, BidirRecurrentComposeAll, BidirRecurrentComposeFirstLast, BidirRecurrentComposeCenter
from net.extend_data_modules import ExtendDataFlow, ExtendDataFocusDist, ExtendDataConstant, ExtendDataFov, ExtendDataMultiple
from net.ddff_net_har import DDFFNetHar


def unet_pair_gen(enc_sizes, dec_sizes):
    return lambda: UNetEncoder(enc_sizes), lambda: UNetDecoder(dec_sizes, enc_sizes)


def create_train_setup(key):
    return train_setups[key]()


train_setups = {
    "ddffnet_dummy": lambda: TrainSetupDepth(
        dataset="s7",
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: DDFFNetDummy(focal_stack_size=4, use_scoring=True)),

    "ddffnet_dummy2": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: DDFFNetDummy(focal_stack_size=4, use_scoring=True)
    ),

    "ddffnet_coc_dummy": lambda: TrainSetupCoCDepthNet(
        dataset="s7",
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: DDFFNetCoCDummy(focal_stack_size=4, use_scoring=False)),

    "ddffnet_fgbg_coc_dummy": lambda: TrainSetupCoCFgbgNet(
        dataset="s7",
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: DDFFNetFgbgCoCDummy(focal_stack_size=4, use_scoring=False)),

    "poolnet_fgbg_coc_2dec_dummy": lambda: TrainSetupCoCFgbgNet(
        dataset="s7",
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: FgbgCocMultiDecoderPoolNet(
            enc_sizes=[4, 6, 8, 16, 24],
            dec_sizes=[24, 16, 12,  6,  6],
            final_conv_sizes=[4, 4],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_fgbg_coc_2net_dummy": lambda: TrainSetupCoCFgbgNet(
        dataset="s7",
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: ParallelNet([PoolNet(
            enc_sizes=[4, 6, 8, 16, 24],
            dec_sizes=[24, 16, 12,  6,  6],
            final_conv_sizes=[4, 4],
            bn_eps=1e-4,
            dec_pool_layers=False
        ) for _ in range(2)])
    ),

    "poolnet_fgbg_coc_2dec_masked_dummy": lambda: TrainSetupCoCFgbgNet(
        dataset="s7",
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0, mask_loss_type=True,
        model_gen=lambda: FgbgCocMultiDecoderPoolNet(
            enc_sizes=[4, 6, 8, 16, 24],
            dec_sizes=[24, 16, 12,  6,  6],
            final_conv_sizes=[4, 4],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_fgbg_coc_layered_dummy": lambda: TrainSetupCoCFgbgNet(
        dataset="s7",
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: CoCFgbgLayeredPoolNet(
            enc_sizes=[4, 6, 8, 16, 24],
            dec_sizes=[24, 16, 12,  6,  6],
            final_conv_sizes=[4, 4],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_coc_dummy": lambda: TrainSetupCoC(
        dataset="s7", mask_loss_type=True,
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        select_focus_dists=[0.1, 0.13, 0.2167, 0.45],
        num_data_threads=1,
        model_gen=lambda: PoolNet(
            enc_sizes=[4, 6, 8, 16, 24],
            dec_sizes=[24, 16, 12,  6,  6],
            final_conv_sizes=[4, 4],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_dummy": lambda: TrainSetupDepth(
        dataset="s7",
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: PoolNet(
            enc_sizes=[4, 6, 8, 16, 24],
            dec_sizes=[24, 16, 12,  6,  6],
            final_conv_sizes=[4, 4],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_addchannel_dummy": lambda: TrainSetupDepth(
        dataset="s7",
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: PoolNet(
            in_channels=4,
            enc_sizes=[1, 1, 1, 1, 1],
            dec_sizes=[1, 1, 1, 1, 1],
            final_conv_sizes=[1, 1, 1],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_signedcoc_dummy": lambda: TrainSetupSignedCoC(
        dataset="s7",
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: PoolNet(
            enc_sizes=[2, 3, 4, 8, 12],
            dec_sizes=[12, 8, 6,  3,  3],
            final_conv_sizes=[2, 2],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "sliding_wnd_poolnet_signedcoc_dummy": lambda: TrainSetupSignedCoC(
        dataset="s7",
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=2, sample_size=25, sample_skip=0,
        target_indices=range(2, 23),
        model_gen=lambda: SlidingWindowNet(net=PoolNet(
            enc_sizes=[2, 3, 4, 8, 12],
            dec_sizes=[12, 8, 6,  3,  3],
            final_conv_sizes=[2, 2],
            bn_eps=1e-4,
            dec_pool_layers=False,
        ), wnd_size=3, reduce=MaxReduce())
    ),

    "ddffnet_lstm_dummy": lambda: TrainSetupDepth(
        dataset="s7",
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=8, sample_size=10, sample_skip=0,
        model_gen=lambda: DDFFNetDummy(focal_stack_size=10, use_scoring=True)),

    "poolnet_consecnet_coc_fgbg_dummy": lambda: TrainSetupCoCFgbgNet(
        dataset="s7",
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        include_input_channel="focus_dist",
        select_focus_dists=[0.1, 0.2167, 0.33, 0.45],
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=in_channels,
            enc_sizes=[4, 6, 8, 16, 24],
            dec_sizes=[24, 16, 12,  6,  6],
            final_conv_sizes=[4, 4],
            bn_eps=1e-4,
            dec_pool_layers=False)
            for in_channels in [3, 5]]
        )
    ),

    "aernn_dummy": lambda: TrainSetupDepth(
        dataset="s7",
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=8, sample_size=5, sample_skip=0,
        model_gen=lambda: AERNN(*UNet.create_encoder_decoder([4, 5, 8, 16]),
                                rnn_dim=32,
                                rnn_num_layers=1,
                                rnn_embedding_type="fc",
                                fs_size=5,
                                ccx_reduce_mode="ccx_last")
    ),

    "vggrnn_dummy": lambda: TrainSetupDepth(
        dataset="s7",
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=8, sample_size=5, sample_skip=0,
        select_focus_dists=[0.1, 0.13, 0.2167, 0.45],
        model_gen=lambda: AERNN(*VGGAE.create_encoder_decoder([4, 6, 8, 10, 12]),
                                rnn_dim=32,
                                rnn_num_layers=1,
                                rnn_embedding_type="fc",
                                fs_size=5,
                                ccx_reduce_mode="ccx_last")
    ),

    "consec_predepth_dummy": lambda: TrainSetupCoCDepthNet(
        dataset="s7",
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        is_predepth_net=True,
        include_input_channel="focus_dist",
        model_gen=lambda: PreDepthConsecNet(nets=[PoolNet(
            in_channels=in_channels,
            enc_sizes=[4, 6, 8, 16, 24],
            dec_sizes=[24, 16, 12, 6, 6],
            final_conv_sizes=[4, 4],
            bn_eps=1e-4,
            dec_pool_layers=False)
            for in_channels in [4, 5]], net_input_indices=[range(4), range(3)]
        )
    ),

    "recur_ae_dummy": lambda: TrainSetupDepth(
        dataset="s7",
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=8, sample_size=5, sample_skip=0,
        select_focus_dists=[0.1, 0.13, 0.2167, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[2, 3, 4, 5])
    ),

    "poolnet_coc_flow_dummy": lambda: TrainSetupCoC(
        checkpoint_freq=1, log_img_freq=1, save_imgs=False, num_epochs=1, log_val_freq=1,
        # select_focus_dists=[0.1, 0.13, 0.2167, 0.45],
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=3, sample_size=5, sample_skip=0,
        include_flow=True,
        extend_data_module=ExtendDataFlow(flow_source="disk", flow_to_first="trajectory", warp_images=False),
        model_gen=lambda: PoolNet(
            in_channels=5,
            enc_sizes=[3, 3, 3, 3, 3],
            dec_sizes=[3, 3, 3, 3, 3],
            final_conv_sizes=[3, 3, 3],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_coc_warp_dummy": lambda: TrainSetupCoC(
        checkpoint_freq=1, log_img_freq=1, save_imgs=False, num_epochs=1, log_val_freq=1,
        # select_focus_dists=[0.1, 0.13, 0.2167, 0.45],
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=1, sample_size=5, sample_skip=0,
        include_flow=True,
        extend_data_module=ExtendDataFlow(flow_source="disk", flow_to_first="trajectory", warp_images=True),
        model_gen=lambda: PoolNet(
            in_channels=3,
            enc_sizes=[3, 3, 3, 3, 3],
            dec_sizes=[3, 3, 3, 3, 3],
            final_conv_sizes=[3, 3, 3],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "consec_pool_recurae_pre4coc_depth_dummy": lambda: TrainSetupCoCDepthNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        pretrained_coc=True,
        extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=4,
            enc_sizes=[1, 1, 1, 1, 1],
            dec_sizes=[1, 1, 1, 1, 1],
            final_conv_sizes=[1, 1, 1],
            bn_eps=1e-4,
            dec_pool_layers=False),
            RecurrentAE(in_channels=2, enc_sizes=[1, 1, 1, 1, 1])
        ], net_input_indices=[range(4), [3]]
        )
    ),

    "consec_pool_pool_pre4coc_fgbg_dummy": lambda: TrainSetupCoCFgbgNet(
        checkpoint_freq=5,
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        pretrained_coc=True,
        extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=4,
            enc_sizes=[1, 1, 1, 1, 1],
            dec_sizes=[1, 1, 1, 1, 1],
            final_conv_sizes=[1, 1, 1],
            bn_eps=1e-4,
            dec_pool_layers=False), PoolNet(
            in_channels=2,
            enc_sizes=[1, 1, 1, 1, 1],
            dec_sizes=[1, 1, 1, 1, 1],
            final_conv_sizes=[1, 1, 1],
            bn_eps=1e-4,
            dec_pool_layers=False),
        ], net_input_indices=[range(4), [3]]
        )
    ),

    "consec_pool_recurae_pre4coc_fgbg_dummy": lambda: TrainSetupCoCFgbgNet(
        checkpoint_freq=5,
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        pretrained_coc=True,
        extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=4,
            enc_sizes=[1, 1, 1, 1, 1],
            dec_sizes=[1, 1, 1, 1, 1],
            final_conv_sizes=[1, 1, 1],
            bn_eps=1e-4,
            dec_pool_layers=False),
            RecurrentAE(in_channels=2, enc_sizes=[1, 1, 1, 1, 1])
        ], net_input_indices=[range(4), [3]]
        )
    ),

    "consec_pool_recurae_pre4coc_rgb2_depth_dummy": lambda: TrainSetupCoCDepthNet(
        checkpoint_freq=5,
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        pretrained_coc=True,
        extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=4,
            enc_sizes=[1, 1, 1, 1, 1],
            dec_sizes=[1, 1, 1, 1, 1],
            final_conv_sizes=[1, 1, 1],
            bn_eps=1e-4,
            dec_pool_layers=False),
            RecurrentAE(in_channels=5, enc_sizes=[1, 1, 1, 1, 1])
        ], net_input_indices=[range(4), range(4)]
        )
    ),

    "recur_ae_dummy_ramp": lambda: TrainSetupDepth(
        dataset="s7_5ramp",
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=8, mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.3625, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[1, 1, 1, 1])
    ),

    "recur_ae_dummy_2ramp": lambda: TrainSetupDepth(
        dataset="s7_5ramp",
        checkpoint_freq=5, log_img_freq=1, save_imgs=False,
        lr=1e-4, batch_size=8, mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.3625, 0.45],
        ramp_sample_count=2,
        model_gen=lambda: InputSplitModule(RecurrentAE(enc_sizes=[1, 1, 1, 1]), 2)
    ),

    "recurae_depth_4_old_cocdepth_dummy": lambda: TrainSetupCoCDepthNet(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        checkpoint_freq=5, log_img_freq=1, save_imgs=True,
        model_gen=lambda: AE2Decoder(RecurrentAE, enc_sizes=[1, 1, 1, 1, 1])
    ),

    "recurae_depth_4_randfoc_bigdata_fovdec_2dec_dummy": lambda: TrainSetupCoCDepthNet(
        dataset="s7_randfoc",
        select_rel_indices=np.linspace(0, 1, 4),
        extend_data_module=ExtendDataFov(use_fov_x=True, use_fov_y=True), extend_encoding=True,
        checkpoint_freq=5, log_img_freq=1,
        model_gen=lambda: AE2Decoder(RecurrentAE, enc_sizes=[1, 1, 1, 1, 1], dec_add_input=2)
    ),

    "ddffnew_rnn_fc_cclast_dummy": lambda: TrainSetupDepth(
        checkpoint_freq=5,
        dataset=["s7_5ramp"], mask_loss_type=True,
        lr=1e-4, batch_size=1,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: AERNN(
            *DDFFNetDummy.create_encoder_decoder(),
            fs_size=4, rnn_dim=2, rnn_num_layers=1,
            rnn_embedding_type="fc", ccx_reduce_mode="ccx_last"
        )
    ),

    "ddffnew_rnn_stack_ccconv_cocdepth_dummy": lambda: TrainSetupCoCDepthNet(
        checkpoint_freq=5,
        dataset="s7_5ramp", mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        batch_size=1,
        model_gen=lambda: AERNN(
            *DDFFNetDummy.create_encoder_decoder(two_dec=True),
            fs_size=4, rnn_num_layers=2,
            rnn_embedding_type="stack", ccx_reduce_mode="ccx_conv")
    ),

    "cvpr_recurrecur_rgb_rgbcoc_dummy": lambda: TrainSetupCoCDepthNet(
        dataset="s7_5ramp", mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: ConsecNet(
            nets=[
                RecurrentAE(enc_sizes=[1] * 5, output_all=True),
                RecurrentAE(enc_sizes=[1] * 5, in_channels=4)
            ],
            net_input_indices=[range(3), range(3)]
        )
    ),

    # "ddffnet": TrainSetupDDFF(),
    # "ddffnet_dummy_test": TrainSetupDDFFDummy(checkpoint_freq=2, num_epochs=5),

    # tests 1
    # "ddffnet_din_room": TrainSetupBlender(lr=1e-4, batch_size=6, num_epochs=10000000),
    # "ddffnet_din_room_small_batch": TrainSetupBlender(lr=1e-4, batch_size=3, num_epochs=10000000),
    # "ddffnet_din_room_dummy": TrainSetupBlenderDummy(lr=1e-4, batch_size=1, num_epochs=5),
    # "ddfflstm_all_din_room": TrainSetupBlenderLSTM(lr=1e-4, batch_size=3, lstm_last_frame_only=False),
    # "ddfflstm_last_din_room": TrainSetupBlenderLSTM(lr=1e-4, batch_size=3, lstm_last_frame_only=True)

    # tests 2


    "cocnet": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: CoCDepthEncShareNet(
            enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
            dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
        )
    ),

    "cocnet_bn": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        model_gen=lambda: CoCDepthEncShareNet(
            enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
            dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
            bn_eps=1e-4
        )
    ),

    "cocnet_big_bn": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        model_gen=lambda: CoCDepthEncShareNet(
            enc_sizes=PoolNetEncoder.dft_sizes,
            dec_sizes=PoolNetDecoder.dft_sizes,
            bn_eps=1e-4
        )
    ),

    "cocnet_all_bn": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: CoCDepthEncShareNet(
            enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
            dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
            bn_eps=1e-4,
            dec_pool_layers=[True, True, True, True, False]
        )
    ),

    "cocnet_all_bn_big": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=2, sample_size=4, sample_skip=0,
        model_gen=lambda: CoCDepthEncShareNet(
            enc_sizes=PoolNetEncoder.dft_sizes,
            dec_sizes=PoolNetDecoder.dft_sizes,
            bn_eps=1e-4,
            dec_pool_layers=[True, True, True, True, False]
        )
    ),

    "cocnet_no_pool": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=2, sample_size=4, sample_skip=0,
        model_gen=lambda: CoCDepthEncShareNet(
            enc_sizes=PoolNetEncoder.dft_sizes,
            dec_sizes=PoolNetDecoder.dft_sizes,
            bn_eps=1e-4,
            dec_pool_layers=[False, False, False, False, False]
        )
    ),

    "cocnet_last_pool": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=3, sample_size=4, sample_skip=0,
        model_gen=lambda: CoCDepthEncShareNet(
            enc_sizes=PoolNetEncoder.dft_sizes,
            dec_sizes=PoolNetDecoder.dft_sizes,
            bn_eps=1e-4,
            dec_pool_layers=[False, False, False, False, True]
        )
    ),



    "cocnet_all": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: CoCDepthEncShareNet(
            enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
            dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
            dec_pool_layers=[True, True, True, True, False]
        )
    ),

    "cocnet_all_nopool": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: CoCDepthEncShareNet(
            enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
            dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
            bn_eps=1e-4,
            dec_pool_layers=[False, False, False, False, False]
        )
    ),

    "cocnet_all_nopool_allinfocus": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0, use_allinfocus=True,
        model_gen=lambda: CoCDepthEncShareNet(
            enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
            dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
            bn_eps=1e-4,
            dec_pool_layers=[False, False, False, False, False]
        )
    ),

    "cocnet8_all": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=0,
        model_gen=lambda: CoCDepthEncShareNet(
            enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
            dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
            dec_pool_layers=[True, True, True, True, False]
        )
    ),

    "cocnet8_all_bn": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=2, sample_size=8, sample_skip=0,
        model_gen=lambda: CoCDepthEncShareNet(
            enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
            dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
            bn_eps=1e-4,
            dec_pool_layers=[True, True, True, True, False]
        )
    ),

    "cocnet8_all_nopool": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=0,
        model_gen=lambda: CoCDepthEncShareNet(
            enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
            dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
            bn_eps=1e-4,
            dec_pool_layers=[False, False, False, False, False]
        )
    ),

    "cocnet8": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=0,
        model_gen=lambda: CoCDepthEncShareNet(
            enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
            dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
        )
    ),

    "cocnet8_nopool": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=0,
        model_gen=lambda: CoCDepthEncShareNet(
            enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
            dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
            dec_pool_layers=[False, False, False, False, False]
        )
    ),



    "cocnet_all_nopool_layered": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: CoCDepthLayeredNet(
            enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
            dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
            bn_eps=1e-4,
            dec_pool_layers=[False, False, False, False, False]
        )
    ),

    "cocnet_all_big_nopool_layered": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: CoCDepthLayeredNet(
            enc_sizes=PoolNetEncoder.dft_sizes,
            dec_sizes=PoolNetDecoder.dft_sizes,
            bn_eps=1e-4,
            dec_pool_layers=[False, False, False, False, False]
        )
    ),

    "cocnet_all_big_nopool_layered_allinfocus": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0, use_allinfocus=True,
        model_gen=lambda: CoCDepthLayeredNet(
            enc_sizes=PoolNetEncoder.dft_sizes,
            dec_sizes=PoolNetDecoder.dft_sizes,
            bn_eps=1e-4,
            dec_pool_layers=[False, False, False, False, False]
        )
    ),

    "poolnet_nopool_layered": lambda: TrainSetupDepth(
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: PoolNet(
            enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
            dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
            bn_eps=1e-4,
            dec_pool_layers=[False, False, False, False, True]
        )
    ),

    "poolnet_nopool_layered_allinfocus": lambda: TrainSetupDepth(
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0, use_allinfocus=True,
        model_gen=lambda: PoolNet(
            enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
            dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
            bn_eps=1e-4,
            dec_pool_layers=[False, False, False, False, True]
        )
    ),

    "poolnet_all_nopool_layered": lambda: TrainSetupDepth(
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: PoolNet(
            enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
            dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
            bn_eps=1e-4,
            dec_pool_layers=[False, False, False, False, False]
        )
    ),

    "poolnet_all_big_nopool_layered": lambda: TrainSetupDepth(
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: PoolNet(
            enc_sizes=PoolNetEncoder.dft_sizes,
            dec_sizes=PoolNetDecoder.dft_sizes,
            bn_eps=1e-4,
            dec_pool_layers=[False, False, False, False, False]
        )
    ),


    "ddffnet": lambda: TrainSetupDDFFBlender(lr=1e-4, batch_size=2, sample_size=8, ddff_scoring="last"),

    "ddffnet_lstm_stack8": lambda: TrainSetupDDFFRNN(
        lr=1e-4, batch_size=2, sample_size=8, dropout=0, rnn_type="lstm",
        rnn_embedding="stack", rnn_num_layers=2, rnn_dropout=0),


    # pool encode [64, 96, 128, 256, 384]
    # pool decode [384, 256, 192, 96, 96]

    # pool encode/2 [32, 48, 64, 128, 192]
    # pool decode/2 [192, 128, 96, 48, 48]

    "cocnet_all_nopool_sandwitch2": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: ConcatNetStackSandwitch(
            *unet_pair_gen([32, 48], [48, 48]),
            CoCDepthEncShareNet(
                in_channels=48,
                out_channels=48,
                enc_sizes=[64, 128, 192],
                dec_sizes=[192, 128, 96],
                bn_eps=1e-4,
                dec_pool_layers=False
            ),
            num_frames=4,
            num_decoder=2
        )
    ),

    "cocnet_all_big_nopool_sandwitch2": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        model_gen=lambda: ConcatNetStackSandwitch(
            *unet_pair_gen([64, 96], [96, 96]),
            CoCDepthEncShareNet(
                in_channels=96,
                out_channels=96,
                enc_sizes=[128, 256, 384],
                dec_sizes=[384, 256, 192],
                bn_eps=1e-4,
                dec_pool_layers=False
            ),
            num_frames=4,
            num_decoder=2
        )
    ),

    "poolnet_nopool_layered_sandwitch2": lambda: TrainSetupDepth(
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: ConcatNetStackSandwitch(
            *unet_pair_gen([32, 48], [48, 48]),
            PoolNet(
                in_channels=48,
                out_channels=48,
                enc_sizes=[64, 128, 192],
                dec_sizes=[192, 128, 96],
                bn_eps=1e-4,
                dec_pool_layers=False
            ),
            num_frames=4
        )
    ),



    "poolnet_nopool_layered_sandwitch2_1more": lambda: TrainSetupDepth(
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        model_gen=lambda: ConcatNetStackSandwitch(
            *unet_pair_gen([32, 48], [48, 48]),
            PoolNet(
                in_channels=48,
                out_channels=48,
                enc_sizes=[64, 128, 192, 256],
                dec_sizes=[256, 192, 128, 96],
                bn_eps=1e-4,
                dec_pool_layers=False
            ),
            num_frames=4
        )
    ),

    "poolnet_nopool_layered_sandwitch2_2more": lambda: TrainSetupDepth(
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        model_gen=lambda: ConcatNetStackSandwitch(
            *unet_pair_gen([32, 48], [48, 48]),
            PoolNet(
                in_channels=48,
                out_channels=48,
                enc_sizes=[64, 128, 192, 256, 256],
                dec_sizes=[256, 256, 192, 128, 96],
                bn_eps=1e-4,
                dec_pool_layers=False
            ),
            num_frames=4
        )
    ),

    "poolnet_nopool_2morefinal": lambda: TrainSetupDepth(
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        model_gen=lambda: PoolNet(
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),



    "densenet3d_pretrained": lambda: TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=16,
        use_concat=[False, False, True, False], use_transp_conv=True, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=False, color_noise_stddev=None, use_2d_dec=False, load_pretrained=True
    ),

    "densenet3d_scratch": lambda: TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=6,
        use_concat=[False, False, True, False], use_transp_conv=True, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=False, color_noise_stddev=None, load_pretrained=False
    ),

    "densenet3d_pretrained_depth_for_all": lambda: TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=16, depth_for_all=True,
        use_concat=[False, False, True, False], use_transp_conv=True, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=False, color_noise_stddev=None, use_2d_dec=False, load_pretrained=True
    ),

    "densenet3d_scratch_depth_for_all": lambda: TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=6, depth_for_all=True,
        use_concat=[False, False, True, False], use_transp_conv=True, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=False, color_noise_stddev=None, load_pretrained=False
    ),

    "ddffnet_lstm_stack_pretrained": lambda: TrainSetupDDFFRNN(
        lr=1e-4, batch_size=6, sample_size=10, dropout=0, rnn_type="lstm",
        rnn_embedding="stack", rnn_num_layers=2, rnn_dropout=0, weight_decay=0, ccx_reduce_mode="ccx_conv",
        load_pretrained=True),

    "ddffnet_lstm_fc_pretrained": lambda: TrainSetupDDFFRNN(
        lr=1e-4, batch_size=6, sample_size=10, dropout=0, rnn_type="lstm",
        rnn_embedding="fc", rnn_num_layers=1, rnn_dropout=0, weight_decay=0, load_pretrained=True),

    "ddffnet_pretrained": lambda: TrainSetupDDFFBlender(batch_size=6, sample_size=10, dropout=0,
                                                        ddff_scoring="last", load_pretrained=True),


    ###################################################################################################################




    "ddffnet_lstm_stack_pretrained_1skip": lambda: TrainSetupDDFFRNN(
        lr=1e-4, batch_size=6, sample_size=10, dropout=0, rnn_type="lstm", sample_skip=1,
        rnn_embedding="stack", rnn_num_layers=2, rnn_dropout=0, weight_decay=0, ccx_reduce_mode="ccx_conv",
        load_pretrained=True),

    "poolnet_nopool_layered_sandwitch2_2more_1skip": lambda: TrainSetupDepth(
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=1,
        model_gen=lambda: ConcatNetStackSandwitch(
            *unet_pair_gen([32, 48], [48, 48]),
            PoolNet(
                in_channels=48,
                out_channels=48,
                enc_sizes=[64, 128, 192, 256, 256],
                dec_sizes=[256, 256, 192, 128, 96],
                bn_eps=1e-4,
                dec_pool_layers=False
            ),
            num_frames=4
        )
    ),

    "cocnet_all_big_nopool_layered_1skip": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=1,
        model_gen=lambda: CoCDepthLayeredNet(
            enc_sizes=PoolNetEncoder.dft_sizes,
            dec_sizes=PoolNetDecoder.dft_sizes,
            bn_eps=1e-4,
            dec_pool_layers=[False, False, False, False, False]
        )
    ),

    "ddffnet_lstm_fc_pretrained_1skip": lambda: TrainSetupDDFFRNN(
        lr=1e-4, batch_size=6, sample_size=10, dropout=0, rnn_type="lstm", sample_skip=1,
        rnn_embedding="fc", rnn_num_layers=1, rnn_dropout=0, weight_decay=0, load_pretrained=True),

    "cocnet_all_nopool_sandwitch2_1skip": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=1,
        model_gen=lambda: ConcatNetStackSandwitch(
            *unet_pair_gen([32, 48], [48, 48]),
            CoCDepthEncShareNet(
                in_channels=48,
                out_channels=48,
                enc_sizes=[64, 128, 192],
                dec_sizes=[192, 128, 96],
                bn_eps=1e-4,
                dec_pool_layers=False
            ),
            num_frames=4,
            num_decoder=2
        )
    ),



    ################################################################################################################

    "poolnet_nopool_layered_sandwitch2_2more_8": lambda: TrainSetupDepth(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=0,
        model_gen=lambda: ConcatNetStackSandwitch(
            *unet_pair_gen([32, 48], [48, 48]),
            PoolNet(
                in_channels=48,
                out_channels=48,
                enc_sizes=[64, 128, 192, 256, 256],
                dec_sizes=[256, 256, 192, 128, 96],
                bn_eps=1e-4,
                dec_pool_layers=False
            ),
            num_frames=8
        )
    ),

    "cocnet_all_big_nopool_layered_8": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=0,
        model_gen=lambda: CoCDepthLayeredNet(
            enc_sizes=PoolNetEncoder.dft_sizes,
            dec_sizes=PoolNetDecoder.dft_sizes,
            bn_eps=1e-4,
            dec_pool_layers=[False, False, False, False, False]
        )
    ),

    "cocnet_all_nopool_sandwitch2_8": lambda: TrainSetupCoCDepthNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=0,
        model_gen=lambda: ConcatNetStackSandwitch(
            *unet_pair_gen([32, 48], [48, 48]),
            CoCDepthEncShareNet(
                in_channels=48,
                out_channels=48,
                enc_sizes=[64, 128, 192],
                dec_sizes=[192, 128, 96],
                bn_eps=1e-4,
                dec_pool_layers=False
            ),
            num_frames=8,
            num_decoder=2
        )
    ),



    "poolnet_nopool_layered_sandwitch2_2more_1": lambda: TrainSetupDepth(
        lr=1e-4, batch_size=8, sample_size=1, sample_skip=0,
        model_gen=lambda: ConcatNetStackSandwitch(
            *unet_pair_gen([32, 48], [48, 48]),
            PoolNet(
                in_channels=48,
                out_channels=48,
                enc_sizes=[64, 128, 192, 256, 256],
                dec_sizes=[256, 256, 192, 128, 96],
                bn_eps=1e-4,
                dec_pool_layers=False
            ),
            num_frames=1
        )
    ),

    "ddffnet_lstm_fc_pretrained_1skip_1": lambda: TrainSetupDDFFRNN(
        lr=1e-4, batch_size=8, sample_size=1, dropout=0, rnn_type="lstm", sample_skip=1,
        rnn_embedding="fc", rnn_num_layers=1, rnn_dropout=0, weight_decay=0, load_pretrained=True),

    "ddffnet_lstm_stack_pretrained_1skip_1": lambda: TrainSetupDDFFRNN(
        lr=1e-4, batch_size=8, sample_size=1, dropout=0, rnn_type="lstm", sample_skip=1,
        rnn_embedding="stack", rnn_num_layers=2, rnn_dropout=0, weight_decay=0, ccx_reduce_mode="ccx_conv",
        load_pretrained=True),


    ###########################################################################################################

    "poolnet_s7": lambda: TrainSetupDepth(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: PoolNet(
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "ddff_lstm_fc_pretrained_s7": lambda: TrainSetupDDFFRNN(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=8, sample_size=10, dropout=0, rnn_type="lstm", sample_skip=0,
        rnn_embedding="fc", rnn_num_layers=1, rnn_dropout=0, weight_decay=0, load_pretrained=True),

    "ddffnet_lstm_stack_s7": lambda: TrainSetupDDFFRNN(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=8, sample_size=10, dropout=0, rnn_type="lstm", sample_skip=0,
        rnn_embedding="stack", rnn_num_layers=2, rnn_dropout=0, weight_decay=0, ccx_reduce_mode="ccx_conv",
        load_pretrained=True),


    "poolnet_fgbg_coc_2dec_s7": lambda: TrainSetupCoCFgbgNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: FgbgCocMultiDecoderPoolNet(
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_fgbg_coc_layered_s7": lambda: TrainSetupCoCFgbgNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: CoCFgbgLayeredPoolNet(
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_fgbg_coc_2net_s7": lambda: TrainSetupCoCFgbgNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: ParallelNet([PoolNet(
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        ) for _ in range(2)])
    ),

    "densenet3d_s7": lambda: TrainSetupDenseNet3DAutoEncoder(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=8, sample_skip=0, depth_for_all=True,
        use_concat=[False, False, True, False], use_transp_conv=True, decoder_conv_kernel_sizes=[3, 3],
        use_2d_dec=False, load_pretrained=True
    ),





    "poolnet_fgbg_coc_2dec_s7_n2": lambda: TrainSetupCoCFgbgNet(
        dataset="s7", mask_loss_type=True, color_noise_stddev=0.01,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: FgbgCocMultiDecoderPoolNet(
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_fgbg_coc_2net_s7_n2": lambda: TrainSetupCoCFgbgNet(
        dataset="s7", mask_loss_type=True, color_noise_stddev=0.01,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: ParallelNet([PoolNet(
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        ) for _ in range(2)])
    ),

    "ddffnet_lstm_stack_s7_n2": lambda: TrainSetupDDFFRNN(
        dataset="s7", mask_loss_type=True, color_noise_stddev=0.01,
        lr=1e-4, batch_size=8, sample_size=10, dropout=0, rnn_type="lstm", sample_skip=0,
        rnn_embedding="stack", rnn_num_layers=2, rnn_dropout=0, weight_decay=0, ccx_reduce_mode="ccx_conv",
        load_pretrained=True),

    "cocnet_s7": lambda: TrainSetupCoCDepthNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: CoCDepthEncShareNet(
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "cocnet_s7_n2": lambda: TrainSetupCoCDepthNet(
        dataset="s7", mask_loss_type=True, color_noise_stddev=0.01,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: CoCDepthEncShareNet(
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_signedcoc_s7": lambda: TrainSetupSignedCoC(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: PoolNet(
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_signedcoc_s7_n2": lambda: TrainSetupSignedCoC(
        dataset="s7", mask_loss_type=True, color_noise_stddev=0.01,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0,
        model_gen=lambda: PoolNet(
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),



    ###########################################################################################################

    "poolnet_fgbg_coc_2dec_s7_n2_fdc": lambda: TrainSetupCoCFgbgNet(
        dataset="s7", mask_loss_type=True, color_noise_stddev=0.01,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0, include_input_channel="focus_dist",
        model_gen=lambda in_channels: FgbgCocMultiDecoderPoolNet(
            in_channels=in_channels,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_fgbg_coc_2net_s7_n2_fdc": lambda: TrainSetupCoCFgbgNet(
        dataset="s7", mask_loss_type=True, color_noise_stddev=0.01,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0, include_input_channel="focus_dist",
        model_gen=lambda in_channels: ParallelNet([PoolNet(
            in_channels=in_channels,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        ) for _ in range(2)])
    ),

    "cocnet_s7_fdc": lambda: TrainSetupCoCDepthNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0, include_input_channel="focus_dist",
        model_gen=lambda in_channels: CoCDepthEncShareNet(
            in_channels=in_channels,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "cocnet_s7_n2_fdc": lambda: TrainSetupCoCDepthNet(
        dataset="s7", mask_loss_type=True, color_noise_stddev=0.01,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0, include_input_channel="focus_dist",
        model_gen=lambda in_channels: CoCDepthEncShareNet(
            in_channels=in_channels,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_signedcoc_s7_fdc": lambda: TrainSetupSignedCoC(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0, include_input_channel="focus_dist",
        model_gen=lambda in_channels: PoolNet(
            in_channels=in_channels,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_signedcoc_s7_n2_fdc": lambda: TrainSetupSignedCoC(
        dataset="s7", mask_loss_type=True, color_noise_stddev=0.01,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0, include_input_channel="focus_dist",
        model_gen=lambda in_channels: PoolNet(
            in_channels=in_channels,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_s7_fdc": lambda: TrainSetupDepth(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0, include_input_channel="focus_dist",
        model_gen=lambda: PoolNet(
            in_channels=4,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_consecnet_coc_fgbg": lambda: TrainSetupCoCFgbgNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        include_input_channel="focus_dist",
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=in_channels,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        ) for in_channels in [3, 5]]
        )
    ),

    "aernn_unet": lambda: TrainSetupDepth(
        dataset="s7",
        lr=1e-4, batch_size=8, sample_size=10, sample_skip=0,
        model_gen=lambda: AERNN(*UNet.create_encoder_decoder([32, 64, 128, 256, 512]),
                                rnn_dim=2048,
                                rnn_num_layers=1,
                                rnn_embedding_type="fc",
                                ccx_reduce_mode="ccx_last")
    ),

    "aernn_unet_fdc": lambda: TrainSetupDepth(
        dataset="s7",
        lr=1e-4, batch_size=8, sample_size=10, sample_skip=0,
        include_input_channel="focus_dist",
        model_gen=lambda: AERNN(*UNet.create_encoder_decoder([32, 64, 128, 256, 512], in_channels=4),
                                rnn_dim=2048,
                                rnn_num_layers=1,
                                rnn_embedding_type="fc",
                                ccx_reduce_mode="ccx_last")
    ),

    # to try
    # poolnet -> lstm consec
    # pre depth as input instead of focus dist
    # coc -> depth directly instead of fgbg

    # try again (not necessary)
    # m96255_poolnet_fgbg_coc_2dec_s7_n2_fdc
    # m96256_poolnet_fgbg_coc_2net_s7_n2_fdc

    "poolnet_consecnet_coc_fgbg_bceweight": lambda: TrainSetupCoCFgbgNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        include_input_channel="focus_dist", use_coc_bce_weight=True,
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=in_channels,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        ) for in_channels in [3, 5]]
        )
    ),

    "sliding_wnd_poolnet_signedcoc": lambda: TrainSetupSignedCoC(
        dataset="s7",
        lr=1e-4, batch_size=1, sample_size=25, sample_skip=0,
        include_input_channel="focus_dist",
        target_indices=range(2, 23),
        model_gen=lambda: SlidingWindowNet(net=PoolNet(
            in_channels=4,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        ), wnd_size=3, reduce=MaxReduce())
    ),

    "sliding_wnd_poolnet_signedcoc_skip1": lambda: TrainSetupSignedCoC(
        dataset="s7",
        lr=1e-4, batch_size=2, sample_size=12, sample_skip=1,
        include_input_channel="focus_dist",
        target_indices=range(2, 10),
        model_gen=lambda: SlidingWindowNet(net=PoolNet(
            in_channels=4,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False,
        ), wnd_size=3, reduce=MaxReduce())
    ),

    # TODO: Try
    "aernn_vgg": lambda: TrainSetupDepth(
        dataset="s7",
        lr=1e-4, batch_size=8, sample_size=10, sample_skip=0,
        model_gen=lambda: AERNN(*VGGAE.create_encoder_decoder([64, 128, 256, 512, 512]),
                                rnn_dim=2048,
                                rnn_num_layers=1,
                                rnn_embedding_type="fc",
                                ccx_reduce_mode="ccx_last")
    ),

    "aernn_vgg_fdc": lambda: TrainSetupDepth(
        dataset="s7",
        lr=1e-4, batch_size=8, sample_size=10, sample_skip=0,
        include_input_channel="focus_dist",
        model_gen=lambda: AERNN(*VGGAE.create_encoder_decoder([64, 128, 256, 512, 512], in_channels=4),
                                rnn_dim=2048,
                                rnn_num_layers=1,
                                rnn_embedding_type="fc",
                                ccx_reduce_mode="ccx_last")
    ),
    
    "poolnet_fgbg_coc_2dec_s7_n2_fdc_bceweight": lambda: TrainSetupCoCFgbgNet(
        dataset="s7", mask_loss_type=True, color_noise_stddev=0.01,
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=0, include_input_channel="focus_dist",
        use_coc_bce_weight=True,
        model_gen=lambda in_channels: FgbgCocMultiDecoderPoolNet(
            in_channels=in_channels,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_consecnet_coc_fgbg_bceweight_4in": lambda: TrainSetupCoCFgbgNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        include_input_channel="focus_dist", use_coc_bce_weight=True,
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=in_channels,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        ) for in_channels in [4, 5]]
        )
    ),



    "aernn_vgg_fdc_halfsize": lambda: TrainSetupDepth(
        dataset="s7",
        lr=1e-4, batch_size=8, sample_size=10, sample_skip=0,
        model_gen=lambda: AERNN(*VGGAE.create_encoder_decoder([32, 64, 128, 256, 256],
                                                              in_channels=3,
                                                              use_skip=[False, False, True, False, False]),
                                rnn_dim=2048,
                                rnn_num_layers=1,
                                rnn_embedding_type="fc",
                                ccx_reduce_mode="ccx_last")
    ),

    "aernn_vgg_fdc_newinit": lambda: TrainSetupDepth(
        dataset="s7",
        lr=1e-4, batch_size=8, sample_size=10, sample_skip=0,
        model_gen=lambda: AERNN(*VGGAE.create_encoder_decoder(in_channels=3,
                                                              use_skip=[False, False, True, False, False]),
                                rnn_dim=2048,
                                rnn_num_layers=1,
                                rnn_embedding_type="fc",
                                ccx_reduce_mode="ccx_last")
    ),

    "aernn_vgg_fdc_ccx_conv": lambda: TrainSetupDepth(
        dataset="s7",
        lr=1e-4, batch_size=8, sample_size=10, sample_skip=0,
        model_gen=lambda: AERNN(*VGGAE.create_encoder_decoder(in_channels=3,
                                                              use_skip=[False, False, True, False, False]),
                                rnn_dim=2048,
                                rnn_num_layers=1,
                                rnn_embedding_type="fc",
                                fs_size=10,
                                ccx_reduce_mode="ccx_conv")
    ),

    "aernn_vgg_fdc_stack": lambda: TrainSetupDepth(
        dataset="s7",
        lr=1e-4, batch_size=8, sample_size=10, sample_skip=0,
        model_gen=lambda: AERNN(*VGGAE.create_encoder_decoder(in_channels=3,
                                                              use_skip=[False, False, True, False, False]),
                                rnn_dim=2048,
                                rnn_num_layers=2,
                                rnn_embedding_type="stack",
                                ccx_reduce_mode="ccx_last")
    ),

    "consec_predepth": lambda: TrainSetupCoCDepthNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        is_predepth_net=True,
        include_input_channel="focus_dist",
        model_gen=lambda: PreDepthConsecNet(nets=[PoolNet(
            in_channels=in_channels,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False)
            for in_channels in [4, 5]], net_input_indices=[range(4), range(3)]
        )
    ),

    "ddff_lstm_fc_pretrained_s7_n2": lambda: TrainSetupDDFFRNN(
        dataset="s7", mask_loss_type=True, color_noise_stddev=0.01,
        lr=1e-4, batch_size=8, sample_size=10, dropout=0, rnn_type="lstm", sample_skip=0,
        rnn_embedding="fc", rnn_num_layers=1, rnn_dropout=0, weight_decay=0, load_pretrained=True),

    "ddff_lstm_fc_notpretrained_s7_n2": lambda: TrainSetupDDFFRNN(
        dataset="s7", mask_loss_type=True, color_noise_stddev=0.01,
        lr=1e-4, batch_size=8, sample_size=10, dropout=0, rnn_type="lstm", sample_skip=0,
        rnn_embedding="fc", rnn_num_layers=1, rnn_dropout=0, weight_decay=0, load_pretrained=False),




    "poolnet_coc_pre3_lin": lambda: TrainSetupCoC(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        select_focus_dists=[0.1, 0.2167, 0.33, 0.45],
        model_gen=lambda: PoolNet(
            in_channels=3,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_coc_pre4_lin": lambda: TrainSetupCoC(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        include_input_channel="focus_dist",
        select_focus_dists=[0.1, 0.2167, 0.33, 0.45],
        model_gen=lambda: PoolNet(
            in_channels=4,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_coc_pre3_nonlin": lambda: TrainSetupCoC(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        select_focus_dists=[0.1, 0.13, 0.2167, 0.45],
        model_gen=lambda: PoolNet(
            in_channels=3,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_coc_pre4_nonlin": lambda: TrainSetupCoC(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        include_input_channel="focus_dist",
        select_focus_dists=[0.1, 0.13, 0.2167, 0.45],
        model_gen=lambda: PoolNet(
            in_channels=4,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_coc_pre3_nonlin8": lambda: TrainSetupCoC(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        include_input_channel="focus_dist",
        select_focus_dists=[0.1, 0.1167, 0.13, 0.167, 0.2, 0.25, 0.33, 0.45],
        model_gen=lambda: PoolNet(
            in_channels=4,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_coc_pre4_nonlin8": lambda: TrainSetupCoC(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        include_input_channel="focus_dist",
        select_focus_dists=[0.1, 0.1167, 0.13, 0.167, 0.2, 0.25, 0.33, 0.45],
        model_gen=lambda: PoolNet(
            in_channels=4,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),



    "recurae_depth": lambda: TrainSetupDepth(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.2167, 0.33, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_coc": lambda: TrainSetupCoC(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.2167, 0.33, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),

    # depth -> coc spelling mistake
    "recurae_depth_slim": lambda: TrainSetupCoC(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.2167, 0.33, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 43, 57, 76, 101])
    ),

    "recurae_depth8": lambda: TrainSetupDepth(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_n2": lambda: TrainSetupDepth(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, color_noise_stddev=0.01,
        select_focus_dists=[0.1, 0.2167, 0.33, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_alt3": lambda: TrainSetupDepth(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.275, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192], alternate=True)
    ),

    "recurae_depth_alt5": lambda: TrainSetupDepth(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.3625, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192], alternate=True)
    ),



    "recurae_depth_all": lambda: TrainSetupDepth(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.2167, 0.33, 0.45],
        model_gen=lambda: BidirRecurrentComposeAll(
            RecurrentAE(enc_sizes=[32, 48, 64, 128, 192]),
            use_hidden=True
        )
    ),

    "recurae_depth_center": lambda: TrainSetupDepth(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.3625, 0.45],
        model_gen=lambda: BidirRecurrentComposeCenter(
            RecurrentAE(enc_sizes=[32, 48, 64, 128, 192]),
            use_hidden=True
        )
    ),

    "recurae_depth_firstlast": lambda: TrainSetupDepth(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.2167, 0.33, 0.45],
        model_gen=lambda: BidirRecurrentComposeFirstLast(
            RecurrentAE(enc_sizes=[32, 48, 64, 128, 192]),
            use_hidden=True
        )
    ),


    "recurae_depth_all2": lambda: TrainSetupDepth(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.2167, 0.33, 0.45],
        model_gen=lambda: BidirRecurrentComposeAll(
            RecurrentAE(enc_sizes=[32, 48, 64, 128, 192]),
            RecurrentAE(enc_sizes=[32, 48, 64, 128, 192]),
        )
    ),

    "recurae_depth_center2": lambda: TrainSetupDepth(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.3625, 0.45],
        model_gen=lambda: BidirRecurrentComposeCenter(
            RecurrentAE(enc_sizes=[32, 48, 64, 128, 192]),
            RecurrentAE(enc_sizes=[32, 48, 64, 128, 192]),
        )
    ),

    "recurae_depth_firstlast2": lambda: TrainSetupDepth(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.2167, 0.33, 0.45],
        model_gen=lambda: BidirRecurrentComposeFirstLast(
            RecurrentAE(enc_sizes=[32, 48, 64, 128, 192]),
            RecurrentAE(enc_sizes=[32, 48, 64, 128, 192]),
        )
    ),



    "recurae_coc_all2": lambda: TrainSetupCoC(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.2167, 0.33, 0.45],
        model_gen=lambda: BidirRecurrentComposeAll(
            RecurrentAE(enc_sizes=[32, 48, 64, 128, 192]),
            RecurrentAE(enc_sizes=[32, 48, 64, 128, 192]),
        )
    ),




    "recurae_coc_all2_first12": lambda: TrainSetupCoC(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.13, 0.183, 0.3],
        model_gen=lambda: BidirRecurrentComposeAll(
            RecurrentAE(enc_sizes=[32, 48, 64, 128, 192]),
            RecurrentAE(enc_sizes=[32, 48, 64, 128, 192]),
        )
    ),

    "poolnet_coc_pre3_first12": lambda: TrainSetupCoC(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        select_focus_dists=[0.1, 0.13, 0.183, 0.3],
        model_gen=lambda: PoolNet(
            in_channels=3,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_coc_pre4_first12": lambda: TrainSetupCoC(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        include_input_channel="focus_dist",
        select_focus_dists=[0.1, 0.13, 0.183, 0.3],
        model_gen=lambda: PoolNet(
            in_channels=4,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),


    "recurae_coc_all2_first8": lambda: TrainSetupCoC(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1167, 0.15, 0.2167],
        model_gen=lambda: BidirRecurrentComposeAll(
            RecurrentAE(enc_sizes=[32, 48, 64, 128, 192]),
            RecurrentAE(enc_sizes=[32, 48, 64, 128, 192]),
        )
    ),

    "poolnet_coc_pre3_first8": lambda: TrainSetupCoC(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        select_focus_dists=[0.1, 0.1167, 0.15, 0.2167],
        model_gen=lambda: PoolNet(
            in_channels=3,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_coc_pre4_first8": lambda: TrainSetupCoC(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        include_input_channel="focus_dist",
        select_focus_dists=[0.1, 0.1167, 0.15, 0.2167],
        model_gen=lambda: PoolNet(
            in_channels=4,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),



    "poolnet_coc_pre3_rand": lambda: TrainSetupCoC(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        model_gen=lambda: PoolNet(
            in_channels=3,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_coc_pre4_rand": lambda: TrainSetupCoC(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: PoolNet(
            in_channels=4,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),





    "poolnet_coc_flow_next": lambda: TrainSetupCoC(
        select_focus_dists=[0.1, 0.13, 0.2167, 0.45],
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        extend_data_module=ExtendDataFlow(flow_to_first=None, warp_images=False),
        model_gen=lambda: PoolNet(
            in_channels=5,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_coc_flow_firstcomp": lambda: TrainSetupCoC(
        select_focus_dists=[0.1, 0.13, 0.2167, 0.45],
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        extend_data_module=ExtendDataFlow(flow_to_first="composite", warp_images=False),
        model_gen=lambda: PoolNet(
            in_channels=5,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_coc_flow_firsttraj": lambda: TrainSetupCoC(
        select_focus_dists=[0.1, 0.13, 0.2167, 0.45],
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        extend_data_module=ExtendDataFlow(flow_to_first="trajectory", warp_images=False),
        model_gen=lambda: PoolNet(
            in_channels=5,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_coc_flow_firstdirect": lambda: TrainSetupCoC(
        select_focus_dists=[0.1, 0.13, 0.2167, 0.45],
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        extend_data_module=ExtendDataFlow(flow_to_first="direct", warp_images=False),
        model_gen=lambda: PoolNet(
            in_channels=5,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_coc_flow_firstdirect_warp": lambda: TrainSetupCoC(
        select_focus_dists=[0.1, 0.13, 0.2167, 0.45],
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        extend_data_module=ExtendDataFlow(flow_to_first="direct", warp_images=True),
        model_gen=lambda: PoolNet(
            in_channels=3,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),




    "poolnet_coc_flow_next_2xparams": lambda: TrainSetupCoC(
        select_focus_dists=[0.1, 0.13, 0.2167, 0.45],
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        extend_data_module=ExtendDataFlow(flow_to_first=None, warp_images=False),
        model_gen=lambda: PoolNet(
            in_channels=5,
            enc_sizes=[64, 96, 128, 256, 384],
            dec_sizes=[384, 256, 192, 96, 96],
            final_conv_sizes=[64, 64, 64],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_coc_flow_next_2xparams_disk": lambda: TrainSetupCoC(
        select_focus_dists=[0.1, 0.13, 0.2167, 0.45],
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, include_flow=True,
        extend_data_module=ExtendDataFlow(flow_source="disk", flow_to_first=None, warp_images=False),
        model_gen=lambda: PoolNet(
            in_channels=5,
            enc_sizes=[64, 96, 128, 256, 384],
            dec_sizes=[384, 256, 192, 96, 96],
            final_conv_sizes=[64, 64, 64],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_coc_flow_next_disk": lambda: TrainSetupCoC(
        select_focus_dists=[0.1, 0.13, 0.2167, 0.45],
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, include_flow=True,
        extend_data_module=ExtendDataFlow(flow_source="disk", flow_to_first=None, warp_images=False),
        model_gen=lambda: PoolNet(
            in_channels=5,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_coc_flow_firstcomp_disk": lambda: TrainSetupCoC(
        select_focus_dists=[0.1, 0.13, 0.2167, 0.45],
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, include_flow=True,
        extend_data_module=ExtendDataFlow(flow_source="disk", flow_to_first="composite", warp_images=False),
        model_gen=lambda: PoolNet(
            in_channels=5,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_coc_flow_firstcomp_warp_disk": lambda: TrainSetupCoC(
        select_focus_dists=[0.1, 0.13, 0.2167, 0.45],
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, include_flow=True,
        extend_data_module=ExtendDataFlow(flow_source="disk", flow_to_first="composite", warp_images=True),
        model_gen=lambda: PoolNet(
            in_channels=3,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    # TODO: try
    "poolnet_coc_flow_firsttraj_disk": lambda: TrainSetupCoC(
        select_focus_dists=[0.1, 0.13, 0.2167, 0.45],
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, include_flow=True,
        extend_data_module=ExtendDataFlow(flow_source="disk", flow_to_first="trajectory", warp_images=False),
        model_gen=lambda: PoolNet(
            in_channels=5,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),




    "consec_pool_pool_pre4coc_depth": lambda: TrainSetupCoCDepthNet(
        checkpoint_freq=5,
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        pretrained_coc=True,
        extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=in_channels,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False)
            for in_channels in [4, 2]], net_input_indices=[range(4), [3]]
        )
    ),

    "consec_pool_pool_pre4coc_rgb2_depth": lambda: TrainSetupCoCDepthNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        pretrained_coc=True,
        extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=in_channels,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False)
            for in_channels in [4, 5]], net_input_indices=[range(4), range(4)]
        )
    ),

    "consec_pool_recurae_pre4coc_depth": lambda: TrainSetupCoCDepthNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        pretrained_coc=True,
        extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=4,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False),
            RecurrentAE(in_channels=2, enc_sizes=[32, 48, 64, 128, 192])
        ], net_input_indices=[range(4), [3]]
        )
    ),

    "consec_pool_recurae_pre4coc_rgb2_depth": lambda: TrainSetupCoCDepthNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        pretrained_coc=True,
        extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=4,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False),
            RecurrentAE(in_channels=5, enc_sizes=[32, 48, 64, 128, 192])
        ], net_input_indices=[range(4), range(4)]
        )
    ),

    "consec_pool_recurae_center_pre4coc_depth": lambda: TrainSetupCoCDepthNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        pretrained_coc=True,
        extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=4,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False),
            BidirRecurrentComposeCenter(
                RecurrentAE(in_channels=2, enc_sizes=[32, 48, 64, 128, 192]),
                RecurrentAE(in_channels=2, enc_sizes=[32, 48, 64, 128, 192]),
            )
        ], net_input_indices=[range(4), [3]]
        )
    ),

    "consec_pool_recurae_center_pre4coc_rgb2_depth": lambda: TrainSetupCoCDepthNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        pretrained_coc=True,
        extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=4,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False),
            BidirRecurrentComposeCenter(
                RecurrentAE(in_channels=5, enc_sizes=[32, 48, 64, 128, 192]),
                RecurrentAE(in_channels=5, enc_sizes=[32, 48, 64, 128, 192]),
            )
        ], net_input_indices=[range(4), range(4)]
        )
    ),



    "consec_pool_pool_pre4coc_fgbg": lambda: TrainSetupCoCFgbgNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        pretrained_coc=True,
        extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=in_channels,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False)
            for in_channels in [4, 2]], net_input_indices=[range(4), [3]]
        )
    ),

    "consec_pool_pool_pre4coc_rgb2_fgbg": lambda: TrainSetupCoCFgbgNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        pretrained_coc=True,
        extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=in_channels,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False)
            for in_channels in [4, 5]], net_input_indices=[range(4), range(4)]
        )
    ),

    "consec_pool_recurae_pre4coc_fgbg": lambda: TrainSetupCoCFgbgNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        pretrained_coc=True,
        extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=4,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False),
            RecurrentAE(in_channels=2, enc_sizes=[32, 48, 64, 128, 192])
        ], net_input_indices=[range(4), [3]]
        )
    ),

    "consec_pool_recurae_pre4coc_rgb2_fgbg": lambda: TrainSetupCoCFgbgNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        pretrained_coc=True,
        extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=4,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False),
            RecurrentAE(in_channels=5, enc_sizes=[32, 48, 64, 128, 192])
        ], net_input_indices=[range(4), range(4)]
        )
    ),

    "consec_pool_recurae_center_pre4coc_fgbg": lambda: TrainSetupCoCFgbgNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        pretrained_coc=True,
        extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=4,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False),
            BidirRecurrentComposeCenter(
                RecurrentAE(in_channels=2, enc_sizes=[32, 48, 64, 128, 192]),
                RecurrentAE(in_channels=2, enc_sizes=[32, 48, 64, 128, 192]),
            )
        ], net_input_indices=[range(4), [3]]
        )
    ),

    "consec_pool_recurae_center_pre4coc_rgb2_fgbg": lambda: TrainSetupCoCFgbgNet(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, sample_size=4, sample_skip=0,
        pretrained_coc=True,
        extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=4,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False),
            BidirRecurrentComposeCenter(
                RecurrentAE(in_channels=5, enc_sizes=[32, 48, 64, 128, 192]),
                RecurrentAE(in_channels=5, enc_sizes=[32, 48, 64, 128, 192]),
            )
        ], net_input_indices=[range(4), range(4)]
        )
    ),
    



    "recurae_depth_randsample": lambda: TrainSetupDepth(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        sample_size=4,
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_n2_randsample": lambda: TrainSetupDepth(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, color_noise_stddev=0.01,
        sample_size=4,
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_center2_randsample": lambda: TrainSetupDepth(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        sample_size=5,
        model_gen=lambda: BidirRecurrentComposeCenter(
            RecurrentAE(enc_sizes=[32, 48, 64, 128, 192]),
            RecurrentAE(enc_sizes=[32, 48, 64, 128, 192]),
        )
    ),
    
    "recurae_depth_randsample_focdist": lambda: TrainSetupDepth(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        sample_size=4, extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: RecurrentAE(in_channels=4, enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_n2_randsample_focdist": lambda: TrainSetupDepth(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4, color_noise_stddev=0.01,
        sample_size=4, extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: RecurrentAE(in_channels=4, enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_center2_randsample_focdist": lambda: TrainSetupDepth(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        sample_size=5, extend_data_module=ExtendDataFocusDist(),
        model_gen=lambda: BidirRecurrentComposeCenter(
            RecurrentAE(in_channels=4, enc_sizes=[32, 48, 64, 128, 192]),
            RecurrentAE(in_channels=4, enc_sizes=[32, 48, 64, 128, 192]),
        )
    ),



    "recurae_depth_3": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.275, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_5": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.3625, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_3_n2": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4, color_noise_stddev=0.01,
        select_focus_dists=[0.1, 0.275, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_5_n2": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4, color_noise_stddev=0.01,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.3625, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),



    "poolnet_coc_pre3_rand_ramp3": lambda: TrainSetupCoC(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.275, 0.45],
        model_gen=lambda: PoolNet(
            in_channels=3,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_coc_pre3_rand_ramp5": lambda: TrainSetupCoC(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.3625, 0.45],
        model_gen=lambda: PoolNet(
            in_channels=3,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),

    "poolnet_coc_pre3_rand_ramp5_n2": lambda: TrainSetupCoC(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4, color_noise_stddev=0.01,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.3625, 0.45],
        model_gen=lambda: PoolNet(
            in_channels=3,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            bn_eps=1e-4,
            dec_pool_layers=False
        )
    ),


    "recurae_depth_4": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_4_randfoc_fovrecurae_depth_4_n2": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4, color_noise_stddev=0.01,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),



    "recurae_depth_fixed_frames": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.3625, 0.45],
        fixed_frame_indices=range(0, 25, 6),
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_fixed_ramp_idx": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.3625, 0.45],
        fixed_ramp_idx=0,
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),


    "recurae_depth_fixed_frames_single_test": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.3625, 0.45],
        fixed_frame_indices=range(0, 25, 6),
        model_gen=lambda: InputSubsetModule(RecurrentAE(enc_sizes=[32, 48, 64, 128, 192]), [-1])
    ),


    "recurae_depth_4_cocdepth": lambda: TrainSetupCoCDepthNet(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: AE2Decoder(RecurrentAE, enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_4_cocdepth_samedec": lambda: TrainSetupCoCDepthNet(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: OutputSplitModule(RecurrentAE, enc_sizes=[32, 48, 64, 128, 192])
    ),





    "recurae_depth_4_big": lambda: TrainSetupDepth(
        dataset=["s7_5ramp", "s7_bigger"], mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_4_big_cocdepth": lambda: TrainSetupCoCDepthNet(
        dataset=["s7_5ramp", "s7_bigger"], mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: AE2Decoder(RecurrentAE, enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_4_big_focdist": lambda: TrainSetupDepth(
        dataset=["s7_5ramp", "s7_bigger"], mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        extend_data_module=ExtendDataConstant(["foc_dist"]),
        model_gen=lambda: RecurrentAE(in_channels=4, enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_4_big_cocdepth_focdist": lambda: TrainSetupCoCDepthNet(
        dataset=["s7_5ramp", "s7_bigger"], mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        extend_data_module=ExtendDataConstant(["foc_dist"]),
        model_gen=lambda: AE2Decoder(RecurrentAE, in_channels=4, enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_4_big_3f_num": lambda: TrainSetupDepth(
        dataset=["s7_5ramp", "s7_bigger"], mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        extend_data_module=ExtendDataConstant(["foc_dist", "focal_length", "f_number"]),
        model_gen=lambda: RecurrentAE(in_channels=4, enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_4_big_cocdepth_3f_num": lambda: TrainSetupCoCDepthNet(
        dataset=["s7_5ramp", "s7_bigger"], mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        extend_data_module=ExtendDataConstant(["foc_dist", "focal_length", "f_number"]),
        model_gen=lambda: AE2Decoder(RecurrentAE, in_channels=4, enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_4_big_cocdepth_3aper": lambda: TrainSetupCoCDepthNet(
        dataset=["s7_5ramp", "s7_bigger"], mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        extend_data_module=ExtendDataConstant(["foc_dist", "focal_length", "aperture"]),
        model_gen=lambda: AE2Decoder(RecurrentAE, in_channels=4, enc_sizes=[32, 48, 64, 128, 192])
    ),




    "recurae_depth_4_big_4fov": lambda: TrainSetupDepth(
        dataset=["s7_5ramp", "s7_bigger"], mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        extend_data_module=ExtendDataConstant(["foc_dist", "focal_length", "f_number", "fov_x"]),
        model_gen=lambda: RecurrentAE(in_channels=4, enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_4_big_cocdepth_4sensor": lambda: TrainSetupCoCDepthNet(
        dataset=["s7_5ramp", "s7_bigger"], mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        extend_data_module=ExtendDataConstant(["foc_dist", "focal_length", "f_number", "sensor_size_x"]),
        model_gen=lambda: AE2Decoder(RecurrentAE, in_channels=4, enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_4_big_fov": lambda: TrainSetupDepth(
        dataset=["s7_5ramp", "s7_bigger"], mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        extend_data_module=ExtendDataConstant(["fov_x"]),
        model_gen=lambda: RecurrentAE(in_channels=4, enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_4_big_fov_focdist": lambda: TrainSetupDepth(
        dataset=["s7_5ramp", "s7_bigger"], mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        extend_data_module=ExtendDataConstant(["fov_x", "foc_dist"]),
        model_gen=lambda: RecurrentAE(in_channels=4, enc_sizes=[32, 48, 64, 128, 192])
    ),



    "recurae_depth_4_old": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_4_old_cocdepth": lambda: TrainSetupCoCDepthNet(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: AE2Decoder(RecurrentAE, enc_sizes=[32, 48, 64, 128, 192])
    ),

    # more feature sizes for recur_ae when training with small+big obj dataset


    "recurae_depth_4_randfoc": lambda: TrainSetupDepth(
        dataset="s7_randfoc",
        select_rel_indices=np.linspace(0, 1, 4),
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_4_randfoc_bigdata": lambda: TrainSetupDepth(
        dataset=["s7_randfoc", "s7_randfoc_bigger"],
        select_rel_indices=np.linspace(0, 1, 4),
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_4_randfoc_bignet": lambda: TrainSetupDepth(
        dataset="s7_randfoc",
        select_rel_indices=np.linspace(0, 1, 4),
        model_gen=lambda: RecurrentAE(enc_sizes=[64, 96, 128, 256, 384])
    ),

    "recurae_depth_4_randfoc_bigdata_bignet": lambda: TrainSetupDepth(
        dataset=["s7_randfoc", "s7_randfoc_bigger"],
        select_rel_indices=np.linspace(0, 1, 4),
        model_gen=lambda: RecurrentAE(enc_sizes=[64, 96, 128, 256, 384])
    ),


    "recurae_depth_4_randfoc_fov": lambda: TrainSetupDepth(
        dataset="s7_randfoc",
        select_rel_indices=np.linspace(0, 1, 4),
        extend_data_module=ExtendDataFov(use_fov_x=True, use_fov_y=True),
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192], in_channels=5)
    ),

    "recurae_depth_4_randfoc_bigdata_fov": lambda: TrainSetupDepth(
        dataset=["s7_randfoc", "s7_randfoc_bigger"],
        select_rel_indices=np.linspace(0, 1, 4),
        extend_data_module=ExtendDataFov(use_fov_x=True, use_fov_y=True),
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192], in_channels=5)
    ),

    "recurae_depth_4_randfoc_bignet_fov": lambda: TrainSetupDepth(
        dataset="s7_randfoc",
        select_rel_indices=np.linspace(0, 1, 4),
        extend_data_module=ExtendDataFov(use_fov_x=True, use_fov_y=True),
        model_gen=lambda: RecurrentAE(enc_sizes=[64, 96, 128, 256, 384], in_channels=5)
    ),

    "recurae_depth_4_randfoc_bigdata_bignet_fov": lambda: TrainSetupDepth(
        dataset=["s7_randfoc", "s7_randfoc_bigger"],
        select_rel_indices=np.linspace(0, 1, 4),
        extend_data_module=ExtendDataFov(use_fov_x=True, use_fov_y=True),
        model_gen=lambda: RecurrentAE(enc_sizes=[64, 96, 128, 256, 384], in_channels=5)
    ),


    "recurae_depth_4_randfoc_focdist": lambda: TrainSetupDepth(
        dataset="s7_randfoc",
        select_rel_indices=np.linspace(0, 1, 4),
        extend_data_module=ExtendDataConstant(["foc_dist"]),
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192], in_channels=4)
    ),


    # *
    "recurae_depth_4_big_bignet": lambda: TrainSetupDepth(
        dataset=["s7_5ramp", "s7_bigger"],
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[64, 96, 128, 256, 384])
    ),

    # test with 2 decoder, test only giving to decoder, test only giving fov_x, test with giving focus_dist

    # *
    "recurae_depth_4_randfoc_2dec": lambda: TrainSetupCoCDepthNet(
        dataset="s7_randfoc",
        select_rel_indices=np.linspace(0, 1, 4),
        model_gen=lambda: AE2Decoder(RecurrentAE, enc_sizes=[32, 48, 64, 128, 192])
    ),


    "recurae_depth_4_randfoc_focdist_fov": lambda: TrainSetupDepth(
        dataset="s7_randfoc",
        select_rel_indices=np.linspace(0, 1, 4),
        extend_data_module=ExtendDataMultiple([
            ExtendDataConstant(["foc_dist"]),
            ExtendDataFov(use_fov_x=True, use_fov_y=True)
        ]),
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192], in_channels=6)
    ),

    "recurae_depth_4_randfoc_focdist_fov_2dec": lambda: TrainSetupCoCDepthNet(
        dataset="s7_randfoc",
        select_rel_indices=np.linspace(0, 1, 4),
        extend_data_module=ExtendDataMultiple([
            ExtendDataConstant(["foc_dist"]),
            ExtendDataFov(use_fov_x=True, use_fov_y=True)
        ]),
        model_gen=lambda: AE2Decoder(RecurrentAE, enc_sizes=[32, 48, 64, 128, 192], in_channels=6)
    ),



    "recurae_depth_4_randfoc_bigdata_2dec": lambda: TrainSetupCoCDepthNet(
        dataset=["s7_randfoc", "s7_randfoc_bigger"],
        select_rel_indices=np.linspace(0, 1, 4),
        model_gen=lambda: AE2Decoder(RecurrentAE, enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_4_randfoc_bigdata_big_2dec": lambda: TrainSetupCoCDepthNet(
        dataset=["s7_randfoc", "s7_randfoc_bigger"],
        select_rel_indices=np.linspace(0, 1, 4),
        model_gen=lambda: AE2Decoder(RecurrentAE, enc_sizes=[64, 96, 128, 256, 384])
    ),

    "recurae_depth_4_randfoc_bigdata_fovdec_2dec": lambda: TrainSetupCoCDepthNet(
        dataset="s7_randfoc",
        select_rel_indices=np.linspace(0, 1, 4),
        extend_data_module=ExtendDataFov(use_fov_x=True, use_fov_y=True), extend_encoding=True,
        model_gen=lambda: AE2Decoder(RecurrentAE, enc_sizes=[32, 48, 64, 128, 192], dec_add_input=2)
    ),


    "recurae_depth_4_randfoc_focdist_fov_morefeat": lambda: TrainSetupDepth(
        dataset="s7_randfoc",
        select_rel_indices=np.linspace(0, 1, 4),
        extend_data_module=ExtendDataMultiple([
            ExtendDataConstant(["foc_dist"]),
            ExtendDataFov(use_fov_x=True, use_fov_y=True)
        ]),
        model_gen=lambda: RecurrentAE(enc_sizes=[64, 96, 128, 256, 384], in_channels=6)
    ),

    "recurae_depth_4_randfoc_focdist_fov_2dec_morefeat": lambda: TrainSetupCoCDepthNet(
        dataset="s7_randfoc",
        select_rel_indices=np.linspace(0, 1, 4),
        extend_data_module=ExtendDataMultiple([
            ExtendDataConstant(["foc_dist"]),
            ExtendDataFov(use_fov_x=True, use_fov_y=True)
        ]),
        model_gen=lambda: AE2Decoder(RecurrentAE, enc_sizes=[64, 96, 128, 256, 384], in_channels=6)
    ),

    "ddff_test": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: DDFFNet(focal_stack_size=4)
        # model_gen=lambda: DDFFNetDummy(focal_stack_size=4, use_scoring=True)
    ),

    "ddff_do_test": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: DDFFNet(focal_stack_size=4, dropout=0.5)
        # model_gen=lambda: DDFFNetDummy(focal_stack_size=4, use_scoring=True)
    ),


    "recurae_depth_4_randfoc_fov_x": lambda: TrainSetupDepth(
        dataset="s7_randfoc",
        select_rel_indices=np.linspace(0, 1, 4),
        extend_data_module=ExtendDataFov(use_fov_x=True, use_fov_y=False),
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192], in_channels=4)
    ),

    "recurae_depth_4_randfoc_fov_x_const": lambda: TrainSetupDepth(
        dataset="s7_randfoc",
        select_rel_indices=np.linspace(0, 1, 4),
        extend_data_module=ExtendDataConstant(["fov_x"]),
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192], in_channels=4)
    ),

    "recurae_depth_4_randfoc_fov_x_old": lambda: TrainSetupDepth(
        dataset="s7_5ramp",
        select_rel_indices=np.linspace(0, 1, 4),
        extend_data_module=ExtendDataFov(use_fov_x=True, use_fov_y=False),
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192], in_channels=4)
    ),


    "recurae_depth_4_near": lambda: TrainSetupDepth(
        dataset="s7_near",
        select_rel_indices=np.linspace(0, 1, 4),
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_4_near_bigger": lambda: TrainSetupDepth(
        dataset=["s7_near", "s7_near_bigger"],
        select_rel_indices=np.linspace(0, 1, 4),
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),




    "ddffhar_do_last": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: DDFFNetHar(focal_stack_size=4, pred_middle=False, dropout=0.5)
        # model_gen=lambda: DDFFNetDummy(focal_stack_size=4, use_scoring=True)
    ),

    "ddffhar_do_middle": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: DDFFNetHar(focal_stack_size=4, pred_middle=True, dropout=0.5)
        # model_gen=lambda: DDFFNetDummy(focal_stack_size=4, use_scoring=True)
    ),

    "ddffhar_nodo_last": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: DDFFNetHar(focal_stack_size=4, pred_middle=False, dropout=0)
        # model_gen=lambda: DDFFNetDummy(focal_stack_size=4, use_scoring=True)
    ),

    "ddffhar_nodo_middle": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: DDFFNetHar(focal_stack_size=4, pred_middle=True, dropout=0)
        # model_gen=lambda: DDFFNetDummy(focal_stack_size=4, use_scoring=True)
    ),




    "ddffnew_do_last": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: DDFFNet(focal_stack_size=4)
    ),


    "ddffnew_do_last_big": lambda: TrainSetupDepth(
        dataset=["s7_5ramp", "s7_bigger"], mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: DDFFNet(focal_stack_size=4)
    ),


    "ddffnew_rnn_fc_cclast": lambda: TrainSetupDepth(
        dataset=["s7_5ramp"], mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: AERNN(
            *DDFFNet.create_encoder_decoder(),
            fs_size=4, rnn_dim=2048, rnn_num_layers=1,
            rnn_embedding_type="fc", ccx_reduce_mode="ccx_last"
        )
    ),

    "ddffnew_rnn_fc_ccconv": lambda: TrainSetupDepth(
        dataset=["s7_5ramp"], mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: AERNN(
            *DDFFNet.create_encoder_decoder(),
            fs_size=4, rnn_dim=2048, rnn_num_layers=1,
            rnn_embedding_type="fc", ccx_reduce_mode="ccx_conv"
        )
    ),

    "ddffnew_rnn_stack_cclast": lambda: TrainSetupDepth(
        dataset=["s7_5ramp"], mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: AERNN(
            *DDFFNet.create_encoder_decoder(),
            fs_size=4, rnn_num_layers=2,
            rnn_embedding_type="stack", ccx_reduce_mode="ccx_last"
        )
    ),

    "ddffnew_rnn_stack_ccconv": lambda: TrainSetupDepth(
        dataset=["s7_5ramp"], mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: AERNN(
            *DDFFNet.create_encoder_decoder(),
            fs_size=4, rnn_num_layers=2,
            rnn_embedding_type="stack", ccx_reduce_mode="ccx_conv"
        )
    ),

    # PoolNet Tests
    # additional foc dist channel (give to 2nd net only)
    # depth direct, 2 dec, consecutive 2 nets
    # consec: rgb_coc, rgb_foccoc, rgb_rgbcoc, rgb_rgbfoccoc | pool_pool, pool_recurae

    # optional 2dec for ddff and ddff_lstm, foc channel for ddff and ddff_lstm (consectutive net with ddff, ddff_lstm)

    "poolnew_dir_noch": lambda: TrainSetupDepth(
        dataset=["s7_5ramp"], mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: PoolNet(
            in_channels=3,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            dec_pool_layers=[False]*4+[True]
        )
    ),

    "poolnew_dir_focdist": lambda: TrainSetupDepth(
        dataset=["s7_5ramp"], mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        extend_data_module=ExtendDataConstant(["foc_dist"]),
        model_gen=lambda: PoolNet(
            in_channels=4,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            dec_pool_layers=[False]*4+[True]
        )
    ),



    "poolpoolnew_rgb_coc": lambda: TrainSetupCoCDepthNet(
        dataset="s7_5ramp", mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=in_channels,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            dec_pool_layers=dec_pool_layers
        ) for in_channels, dec_pool_layers in zip([3, 1], [False, [False]*4+[True]])],
            net_input_indices=[range(3), []]
        )
    ),

    "poolpoolnew_rgb_foccoc": lambda: TrainSetupCoCDepthNet(
        dataset="s7_5ramp", mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        extend_data_module=ExtendDataConstant(["foc_dist"]),
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=in_channels,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            dec_pool_layers=dec_pool_layers
        ) for in_channels, dec_pool_layers in zip([3, 2], [False, [False]*4+[True]])],
            net_input_indices=[range(3), [3]]
        )
    ),

    "poolpoolnew_rgb_rgbcoc": lambda: TrainSetupCoCDepthNet(
        dataset="s7_5ramp", mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=in_channels,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            dec_pool_layers=dec_pool_layers
        ) for in_channels, dec_pool_layers in zip([3, 4], [False, [False]*4+[True]])],
            net_input_indices=[range(3), range(3)]
        )
    ),

    "poolpoolnew_rgb_rgbfoccoc": lambda: TrainSetupCoCDepthNet(
        dataset="s7_5ramp", mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        extend_data_module=ExtendDataConstant(["foc_dist"]),
        model_gen=lambda: ConsecNet(nets=[PoolNet(
            in_channels=in_channels,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            dec_pool_layers=dec_pool_layers
        ) for in_channels, dec_pool_layers in zip([3, 5], [False, [False]*4+[True]])],
            net_input_indices=[range(3), range(4)]
        )
    ),

    "poolnew_dir_noch_2dec": lambda: TrainSetupCoCDepthNet(
        dataset=["s7_5ramp"], mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: AE2Decoder(PoolNet, 
            in_channels=3,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            dec_pool_layers=[False]*4+[True]
        )
    ),

    "poolnew_dir_focdist_2dec": lambda: TrainSetupCoCDepthNet(
        dataset=["s7_5ramp"], mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        extend_data_module=ExtendDataConstant(["foc_dist"]),
        model_gen=lambda: AE2Decoder(PoolNet,
            in_channels=4,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            dec_pool_layers=[False]*4+[True]
        )
    ),

    # train recur_ae for 1, 2, 3, 4, 5 input frames and compare results

    "recurae_depth_numframetest_1": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_numframetest_2": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_numframetest_3": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.275, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_numframetest_5": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.3625, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),


    "poolnew_dir_noch_coc": lambda: TrainSetupCoC(
        dataset=["s7_5ramp"], mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: PoolNet(
            in_channels=3,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32],
            dec_pool_layers=[False] * 4 + [True]
        )
    ),

    "recurae_depth_numframetest_4": lambda: TrainSetupDepth(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192])
    ),

    "recurae_depth_4_cocdepth_focdist": lambda: TrainSetupCoCDepthNet(
        dataset="s7_5ramp", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        extend_data_module=ExtendDataConstant(["foc_dist"]),
        model_gen=lambda: AE2Decoder(RecurrentAE, in_channels=4, enc_sizes=[32, 48, 64, 128, 192])
    ),



    "ddffnew_rnn_stack_ccconv_cocdepth": lambda: TrainSetupCoCDepthNet(
        dataset="s7_5ramp", mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: AERNN(
            *DDFFNet.create_encoder_decoder(two_dec=True),
            fs_size=4, rnn_num_layers=2,
            rnn_embedding_type="stack", ccx_reduce_mode="ccx_conv")
    ),

    "ddffnew_rnn_stack_ccconv_big": lambda: TrainSetupDepth(
        dataset=["s7_5ramp", "s7_bigger"], mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: AERNN(
            *DDFFNet.create_encoder_decoder(),
            fs_size=4, rnn_num_layers=2,
            rnn_embedding_type="stack", ccx_reduce_mode="ccx_conv"
        )
    ),

    "ddffnew_rnn_stack_ccconv_big_cocdepth": lambda: TrainSetupCoCDepthNet(
        dataset=["s7_5ramp", "s7_bigger"], mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: AERNN(
            *DDFFNet.create_encoder_decoder(two_dec=True),
            fs_size=4, rnn_num_layers=2,
            rnn_embedding_type="stack", ccx_reduce_mode="ccx_conv")
    ),

    "ddffnew_rnn_stack_ccconv_cocpred": lambda: TrainSetupCoC(
        dataset=["s7_5ramp"], mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: AERNN(
            *DDFFNet.create_encoder_decoder(),
            fs_size=4, rnn_num_layers=2,
            rnn_embedding_type="stack", ccx_reduce_mode="ccx_conv"
        )
    ),


    "cvpr_recurrecur_rgb_rgbcoc": lambda: TrainSetupCoCDepthNet(
        dataset="s7_5ramp", mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: ConsecNet(
            nets=[
                RecurrentAE(enc_sizes=[32, 48, 64, 128, 192], output_all=True),
                RecurrentAE(enc_sizes=[32, 48, 64, 128, 192], in_channels=4)
            ],
            net_input_indices=[range(3), range(3)]
        )
    ),

    "cvpr_recurrecur_rgb_rgbcoc_large": lambda: TrainSetupCoCDepthNet(
        dataset=["s7_5ramp", "s7_bigger"], mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: ConsecNet(
            nets=[
                RecurrentAE(enc_sizes=[32, 48, 64, 128, 192], output_all=True),
                RecurrentAE(enc_sizes=[32, 48, 64, 128, 192], in_channels=4)
            ],
            net_input_indices=[range(3), range(3)]
        )
    ),

    "cvpr_recur_coc": lambda: TrainSetupCoC(
        dataset="s7_5ramp", mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192], output_all=True)
    ),

    "cvpr_recur_coc_large": lambda: TrainSetupCoC(
        dataset=["s7_5ramp", "s7_bigger"], mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: RecurrentAE(enc_sizes=[32, 48, 64, 128, 192], output_all=True)
    ),
}

"""
    "poolnew_2dec_noch": lambda: TrainSetupDepth(
        dataset=["s7_5ramp"], mask_loss_type=True,
        select_focus_dists=[0.1, 0.1875, 0.275, 0.45],
        model_gen=lambda: PoolNet(
            in_channels=3,
            enc_sizes=[32, 48, 64, 128, 192],
            dec_sizes=[192, 128, 96, 48, 48],
            final_conv_sizes=[32, 32, 32]
        )
    ),
    """


"""recurae_depth_4
    "recurae_depth8": lambda: TrainSetupCoC(
        dataset="s7", mask_loss_type=True,
        lr=1e-4, batch_size=4,
        select_focus_dists=[0.1, 0.1167, 0.13, 0.15],
        model_gen=lambda: BidirRecurrentComposeAll(
            RecurrentAE(enc_sizes=[32, 48, 64, 128, 192]),
            RecurrentAE(enc_sizes=[32, 48, 64, 128, 192]),
        )
    ),"""


"""
    "ddff_lstm_one_enc_skip5": TrainSetupDDFFRNN(
        lr=1e-4, batch_size=3, sample_size=10, dropout=0, sample_skip=5, rnn_dim=2048, rnn_type="lstm"),
    "ddff_lstm_one_enc_skip10": TrainSetupDDFFRNN(
        lr=1e-4, batch_size=3, sample_size=10, dropout=0, sample_skip=10, rnn_dim=2048, rnn_type="lstm"),
    "ddff_lstm_one_enc_skip2_25f_lstm4096": TrainSetupDDFFRNN(
        lr=1e-4, batch_size=3, sample_size=25, dropout=0, sample_skip=2, rnn_dim=4096, rnn_type="lstm"),
    "ddff_lstm_one_enc_skip4_25f_lstm4096": TrainSetupDDFFRNN(
        lr=1e-4, batch_size=3, sample_size=25, dropout=0, sample_skip=4, rnn_dim=4096, rnn_type="lstm"),

    "ddffnet": TrainSetupDDFFBlender(),
    "ddff_lstm_all_enc_skip2_25f_lstm4096": TrainSetupDDFFRNN(
        lr=1e-4, batch_size=3, sample_size=25, dropout=0, sample_skip=2, rnn_dim=4096, rnn_type="lstm"),

    "ddffnet_do5": TrainSetupDDFFBlender(batch_size=6, dropout=0.5),
    "ddff_lstm_all_enc_skip2_25f_lstm4096_do5": TrainSetupDDFFRNN(
        lr=1e-4, batch_size=3, sample_size=25, dropout=0.5, sample_skip=2, rnn_dim=4096, rnn_type="lstm"),

    "ddffnet_lstm_stack": TrainSetupDDFFRNN(
        lr=1e-4, batch_size=3, sample_size=25, dropout=0, sample_skip=2, rnn_type="lstm",
        rnn_embedding="stack", rnn_num_layers=2, rnn_dropout=0),

    "ddffnet_lstm_stack_dob5": TrainSetupDDFFRNN(
            lr=1e-4, batch_size=3, sample_size=25, dropout=0.5, sample_skip=2, rnn_type="lstm",
            rnn_embedding="stack", rnn_num_layers=2, rnn_dropout=0.5),

    "ddff_lstm_all_enc_skip2_25f_lstm4096_wd": TrainSetupDDFFRNN(
        lr=1e-3, batch_size=3, sample_size=25, dropout=0, sample_skip=2, rnn_dim=4096, rnn_type="lstm",
        weight_decay=5e-4),
    "ddffnet_lstm_stack_wd": TrainSetupDDFFRNN(
            lr=5e-4, batch_size=3, sample_size=25, dropout=0.5, sample_skip=2, rnn_type="lstm",
            rnn_embedding="stack", rnn_num_layers=2, rnn_dropout=0.5, weight_decay=5e-4),

    "ddffnet_wd": TrainSetupDDFFBlender(lr=5e-3, batch_size=6, dropout=0.5, weight_decay=5e-4),
    "ddff_lstm_all_enc_skip2_10f_lstm4096_wd": TrainSetupDDFFRNN(
        lr=5e-3, batch_size=3, sample_size=10, dropout=0.5, sample_skip=2, rnn_dim=4096, rnn_type="lstm",
        weight_decay=5e-4),
    "ddffnet_lstm_stack_10f_wd": TrainSetupDDFFRNN(
        lr=5e-4, batch_size=3, sample_size=10, dropout=0.5, sample_skip=2, rnn_type="lstm",
        rnn_embedding="stack", rnn_num_layers=2, rnn_dropout=0.5, weight_decay=5e-4),

    "ddffnet_lstm_fc_10f_skip1": TrainSetupDDFFRNN(
        lr=1e-4, batch_size=6, sample_size=10, dropout=0, sample_skip=1, rnn_type="lstm",
        rnn_embedding="fc", rnn_num_layers=1, rnn_dropout=0, weight_decay=0),
    "ddffnet_lstm_stack_10f_skip1": TrainSetupDDFFRNN(
        lr=1e-4, batch_size=6, sample_size=10, dropout=0, sample_skip=1, rnn_type="lstm",
        rnn_embedding="stack", rnn_num_layers=2, rnn_dropout=0, weight_decay=0),

    "ddffnet_10f_skip1_ccx_conv": TrainSetupDDFFBlender(
            lr=1e-4, batch_size=6, sample_size=10, dropout=0, sample_skip=1,
            weight_decay=0, ddff_scoring="inter/ccx_conv"),
    "ddffnet_lstm_stack_10f_skip1_ccx_conv": TrainSetupDDFFRNN(
        lr=1e-4, batch_size=6, sample_size=10, dropout=0, sample_skip=1, rnn_type="lstm",
        rnn_embedding="stack", rnn_num_layers=2, rnn_dropout=0, weight_decay=0, ccx_reduce_mode="ccx_conv"),

    "ddffnet_lstm_stack_10f_skip1_noccx": TrainSetupDDFFRNN(
            lr=1e-4, batch_size=6, sample_size=10, dropout=0, sample_skip=1, rnn_type="lstm",
            rnn_embedding="stack", rnn_num_layers=2, rnn_dropout=0, weight_decay=0, use_ccx=[False]*5),

    "ddffnet_lstm_stack_10f_skip1_allinfocus": TrainSetupDDFFRNN(
        lr=1e-4, batch_size=6, sample_size=10, dropout=0, sample_skip=1, rnn_type="lstm",
        rnn_embedding="stack", rnn_num_layers=2, rnn_dropout=0, weight_decay=0, use_allinfocus=True),

    "ddffnet_do1": TrainSetupDDFFBlender(batch_size=6, sample_size=10, dropout=0.1, sample_skip=1, ddff_scoring="last"),

    "ddffnet_std": TrainSetupDDFFBlender(batch_size=6, sample_size=10, dropout=0, sample_skip=1,
                                         ddff_scoring="last"),

    "ddffnet_allinfocus": TrainSetupDDFFBlender(batch_size=6, sample_size=10, dropout=0, sample_skip=1,
                                                ddff_scoring="last", use_allinfocus=True),

    "ddffnet_lstm_stack_10f_skip1_allinfocus_ne-2": TrainSetupDDFFRNN(
        lr=1e-4, batch_size=6, sample_size=10, dropout=0, sample_skip=1, rnn_type="lstm",
        rnn_embedding="stack", rnn_num_layers=2, rnn_dropout=0, weight_decay=0, use_allinfocus=True,
        color_noise_stddev=0.01
    ),

    "densenet3d": TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=6, sample_skip=1,
        use_concat=[False, False, True, False], use_transp_conv=True, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=False, color_noise_stddev=None
    ),

    "densenet3d_bil_allcc": TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=8, sample_skip=1,
        use_concat=[True, True, True, False], use_transp_conv=False, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=False, color_noise_stddev=None
    ),

    "densenet3d_b16_bil_allcc": TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=16, sample_skip=1, data_expand=4,
        use_concat=[True, True, True, False], use_transp_conv=False, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=False, color_noise_stddev=None
    ),



    "densenet3d_b16": TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=16, sample_skip=1, data_expand=4,
        use_concat=[False, False, True, False], use_transp_conv=True, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=False, color_noise_stddev=None
    ),

    "densenet3d_b16_bil": TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=16, sample_skip=1, data_expand=4,
        use_concat=[False, False, True, False], use_transp_conv=False, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=False, color_noise_stddev=None
    ),

    "densenet3d_b16_noccx": TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=16, sample_skip=1, data_expand=4,
        use_concat=[False, False, False, False], use_transp_conv=True, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=False, color_noise_stddev=None
    ),

    "densenet3d_b16_allccx": TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=15, sample_skip=1, data_expand=4,
        use_concat=[True, True, True, False], use_transp_conv=True, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=False, color_noise_stddev=None
    ),

    "densenet3d_b16_allinfocus": TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=16, sample_skip=1, data_expand=4,
        use_concat=[False, False, True, False], use_transp_conv=True, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=True, color_noise_stddev=None
    ),



    "densenet3d_b16_2d": TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=16, sample_skip=1, data_expand=4,
        use_concat=[False, False, True, False], use_transp_conv=True, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=False, color_noise_stddev=None, use_2d_dec=True
    ),

    "densenet3d_b16_bil_2d": TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=16, sample_skip=1, data_expand=4,
        use_concat=[False, False, True, False], use_transp_conv=False, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=False, color_noise_stddev=None, use_2d_dec=True
    ),

    "densenet3d_b16_noccx_2d": TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=16, sample_skip=1, data_expand=4,
        use_concat=[False, False, False, False], use_transp_conv=True, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=False, color_noise_stddev=None, use_2d_dec=True
    ),

    "densenet3d_b16_allccx_2d": TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=15, sample_skip=1, data_expand=4,
        use_concat=[True, True, True, False], use_transp_conv=True, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=False, color_noise_stddev=None, use_2d_dec=True
    ),

    "densenet3d_b16_allinfocus_2d": TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=16, sample_skip=1, data_expand=4,
        use_concat=[False, False, True, False], use_transp_conv=True, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=True, color_noise_stddev=None, use_2d_dec=True
    ),



    "densenet3d_b16_depthforall": TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=16, sample_skip=1, data_expand=4,
        use_concat=[False, False, True, False], use_transp_conv=True, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=False, color_noise_stddev=None, depth_for_all=True
    ),

    "densenet3d_b16_bil_depthforall": TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=16, sample_skip=1, data_expand=4,
        use_concat=[False, False, True, False], use_transp_conv=False, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=False, color_noise_stddev=None, depth_for_all=True
    ),

    "densenet3d_b16_noccx_depthforall": TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=16, sample_skip=1, data_expand=4,
        use_concat=[False, False, False, False], use_transp_conv=True, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=False, color_noise_stddev=None, depth_for_all=True
    ),

    "densenet3d_b16_allccx_depthforall": TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=14, sample_skip=1, data_expand=4,
        use_concat=[True, True, True, False], use_transp_conv=True, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=False, color_noise_stddev=None, depth_for_all=True
    ),

    "densenet3d_b16_allinfocus_depthforall": TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=16, sample_skip=1, data_expand=4,
        use_concat=[False, False, True, False], use_transp_conv=True, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=True, color_noise_stddev=None, depth_for_all=True
    ),

    # 25.03.2019 -- end --

    "ddffnet_scratch": TrainSetupDDFFBlender(batch_size=6, sample_size=10, dropout=0, sample_skip=1,
                                             ddff_scoring="last", load_pretrained=False),

    "ddffnet_lstm_stack_scratch": TrainSetupDDFFRNN(
        lr=1e-4, batch_size=6, sample_size=10, dropout=0, sample_skip=1, rnn_type="lstm",
        rnn_embedding="stack", rnn_num_layers=2, rnn_dropout=0, weight_decay=0, ccx_reduce_mode="ccx_conv",
        load_pretrained=False),

    "ddffnet_lstm_fc_scratch": TrainSetupDDFFRNN(
        lr=1e-4, batch_size=6, sample_size=10, dropout=0, sample_skip=1, rnn_type="lstm",
        rnn_embedding="fc", rnn_num_layers=1, rnn_dropout=0, weight_decay=0, load_pretrained=False),

    "ddffnet_lstmx5_stack_scratch": TrainSetupDDFFRNN(
        lr=1e-4, batch_size=6, sample_size=10, dropout=0, sample_skip=1, rnn_type="lstm",
        rnn_embedding="stack", rnn_num_layers=5, rnn_dropout=0, weight_decay=0, ccx_reduce_mode="ccx_conv",
        load_pretrained=False),

    "densenet3d_scratch": TrainSetupDenseNet3DAutoEncoder(
        lr=1e-4, batch_size=6, sample_skip=1,
        use_concat=[False, False, True, False], use_transp_conv=True, decoder_conv_kernel_sizes=[3, 3],
        use_allinfocus=False, color_noise_stddev=None, load_pretrained=False
    ),




    "poolnet": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=1,
        model=PoolNet(enc_sizes=[64, 96, 128, 256, 384],
                      dec_sizes=[384, 256, 192, 96, 96],
                      bn_eps=None,
                      bias=False,
                      act_func=nn.ELU())
    ),

    "poolnet_nobn": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=1,
        model=PoolNet(enc_sizes=[64, 96, 128, 256, 384],
                      dec_sizes=[384, 256, 192, 96, 96],
                      bn_eps=None,
                      bias=False,
                      act_func=nn.ELU())
    ),

    "poolnet_relu": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=1,
        model=PoolNet(enc_sizes=[64, 96, 128, 256, 384],
                      dec_sizes=[384, 256, 192, 96, 96],
                      bn_eps=None,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_4layer": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=1,
        model=PoolNet(enc_sizes=[64, 96, 128, 256],
                      dec_sizes=[256, 192, 96, 96],
                      bn_eps=None,
                      bias=False,
                      act_func=nn.ELU())
    ),

    "poolnet_3layer": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=1,
        model=PoolNet(enc_sizes=[64, 96, 128],
                      dec_sizes=[192, 96, 96],
                      bn_eps=None,
                      bias=False,
                      act_func=nn.ELU())
    ),

    "poolnet_75dim": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=1,
        model=PoolNet(enc_sizes=[48, 72, 96, 192, 288],
                      dec_sizes=[288, 192, 144, 72, 72],
                      bn_eps=None,
                      bias=False,
                      act_func=nn.ELU())
    ),

    "poolnet_50dim": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=1,
        model=PoolNet(enc_sizes=[32, 48, 64, 128, 192],
                      dec_sizes=[192, 128, 96, 48, 48],
                      bn_eps=None,
                      bias=False,
                      act_func=nn.ELU())
    ),



    "poolnet_50dim_relu": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=1,
        model=PoolNet(enc_sizes=[32, 48, 64, 128, 192],
                      dec_sizes=[192, 128, 96, 48, 48],
                      bn_eps=None,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_50dim_relu_bn": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=1,
        model=PoolNet(enc_sizes=[32, 48, 64, 128, 192],
                      dec_sizes=[192, 128, 96, 48, 48],
                      bn_eps=1e-3,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_25dim_relu": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=1,
        model=PoolNet(enc_sizes=[16, 24, 32, 64, 96],
                      dec_sizes=[96, 64, 48, 24, 24],
                      bn_eps=None,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_50dim_relu_sc3": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=3, sample_skip=1,
        model=PoolNet(enc_sizes=[32, 48, 64, 128, 192],
                      dec_sizes=[192, 128, 96, 48, 48],
                      bn_eps=None,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_50dim_relu_ss2": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=2,
        model=PoolNet(enc_sizes=[32, 48, 64, 128, 192],
                      dec_sizes=[192, 128, 96, 48, 48],
                      bn_eps=None,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_50dim_relu_ss3": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=3,
        model=PoolNet(enc_sizes=[32, 48, 64, 128, 192],
                      dec_sizes=[192, 128, 96, 48, 48],
                      bn_eps=None,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_125dim_relu": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=1,
        model=PoolNet(enc_sizes=[8, 12, 16, 32, 48],
                      dec_sizes=[48, 32, 24, 12, 12],
                      bn_eps=None,
                      bias=False,
                      act_func=nn.ReLU())
    ),





    "poolnet_50dim_relu_bn1e4": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=1,
        model=PoolNet(enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
                      dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
                      bn_eps=1e-4,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_50dim_relu_bn1e5": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=1,
        model=PoolNet(enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
                      dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
                      bn_eps=1e-5,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_75dim_relu_bn1e3": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=1,
        model=PoolNet(enc_sizes=PoolNetEncoder.dft_sizes * 3 // 4,
                      dec_sizes=PoolNetDecoder.dft_sizes * 3 // 4,
                      bn_eps=1e-3,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_75dim_relu_bn1e4": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=1,
        model=PoolNet(enc_sizes=PoolNetEncoder.dft_sizes * 3 // 4,
                      dec_sizes=PoolNetDecoder.dft_sizes * 3 // 4,
                      bn_eps=1e-4,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_75dim_relu_bn1e5": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=1,
        model=PoolNet(enc_sizes=PoolNetEncoder.dft_sizes * 3 // 4,
                      dec_sizes=PoolNetDecoder.dft_sizes * 3 // 4,
                      bn_eps=1e-5,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_625dim_relu_bn1e3": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=1,
        model=PoolNet(enc_sizes=PoolNetEncoder.dft_sizes * 5 // 8,
                      dec_sizes=PoolNetDecoder.dft_sizes * 5 // 8,
                      bn_eps=1e-3,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_625dim_relu_bn1e4": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=1,
        model=PoolNet(enc_sizes=PoolNetEncoder.dft_sizes * 5 // 8,
                      dec_sizes=PoolNetDecoder.dft_sizes * 5 // 8,
                      bn_eps=1e-4,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_625dim_relu_bn1e5": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=8, sample_skip=1,
        model=PoolNet(enc_sizes=PoolNetEncoder.dft_sizes * 5 // 8,
                      dec_sizes=PoolNetDecoder.dft_sizes * 5 // 8,
                      bn_eps=1e-5,
                      bias=False,
                      act_func=nn.ReLU())
    ),






    "poolnet_50dim_relu_bn1e4_sc4": TrainSetupPoolNet(
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=1,
        model=PoolNet(enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
                      dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
                      bn_eps=1e-4,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_50dim_relu_bn1e4_sc6": TrainSetupPoolNet(
        lr=1e-4, batch_size=8, sample_size=6, sample_skip=1,
        model=PoolNet(enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
                      dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
                      bn_eps=1e-4,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_50dim_relu_bn1e4_sc12": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=12, sample_skip=1,
        model=PoolNet(enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
                      dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
                      bn_eps=1e-4,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_50dim_relu_bn1e4_sc12_ss0": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=12, sample_skip=0,
        model=PoolNet(enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
                      dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
                      bn_eps=1e-4,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_50dim_relu_bn1e4_sc16": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=16, sample_skip=1,
        model=PoolNet(enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
                      dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
                      bn_eps=1e-4,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_50dim_relu_bn1e4_sc16_ss0": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=12, sample_skip=0,
        model=PoolNet(enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
                      dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
                      bn_eps=1e-4,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_50dim_relu_bn1e4_sc4_ss2": TrainSetupPoolNet(
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=2,
        model=PoolNet(enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
                      dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
                      bn_eps=1e-4,
                      bias=False,
                      act_func=nn.ReLU())
    ),




    "poolnet_50dim_relu_bn1e4_sc12_ss0_allinfocus": TrainSetupPoolNet(
        lr=1e-4, batch_size=4, sample_size=12, sample_skip=0, use_allinfocus=True,
        model=PoolNet(enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
                      dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
                      bn_eps=1e-4,
                      bias=False,
                      act_func=nn.ReLU())
    ),

    "poolnet_50dim_relu_bn1e4_sc4_ss2_allinfocus": TrainSetupPoolNet(
        lr=1e-4, batch_size=8, sample_size=4, sample_skip=2, use_allinfocus=True,
        model=PoolNet(enc_sizes=PoolNetEncoder.dft_sizes * 1 // 2,
                      dec_sizes=PoolNetDecoder.dft_sizes * 1 // 2,
                      bn_eps=1e-4,
                      bias=False,
                      act_func=nn.ReLU())
    ),
    """

"""
   "poolnet_std": TrainSetupPoolNet(
       lr=1e-4, batch_size=8, sample_size=10, sample_skip=1,
       inter_feature_sizes=[64, 128, 256, 512], post_conv_channel_size=1, bn_eps=1e-3, last_relu=False, bias=False
   ),

   "poolnet_less": TrainSetupPoolNet(
       lr=1e-4, batch_size=8, sample_size=10, sample_skip=1,
       inter_feature_sizes=[64, 128, 256], post_conv_channel_size=1, bn_eps=1e-3, last_relu=False, bias=False
   ),

   "poolnet_bias": TrainSetupPoolNet(
       lr=1e-4, batch_size=8, sample_size=10, sample_skip=1,
       inter_feature_sizes=[64, 128, 256, 512], post_conv_channel_size=1, bn_eps=1e-3, last_relu=False, bias=True
   ),

   "poolnet_relu": TrainSetupPoolNet(
       lr=1e-4, batch_size=8, sample_size=10, sample_skip=1,
       inter_feature_sizes=[64, 128, 256, 512], post_conv_channel_size=1, bn_eps=1e-3, last_relu=True, bias=False
   ),

   "poolnet_noskip": TrainSetupPoolNet(
       lr=1e-4, batch_size=8, sample_size=10, sample_skip=0,
       inter_feature_sizes=[64, 128, 256, 512], post_conv_channel_size=1, bn_eps=1e-3, last_relu=False, bias=False
   ),

   "poolnet_nobn": TrainSetupPoolNet(
       lr=1e-4, batch_size=8, sample_size=10, sample_skip=1,
       inter_feature_sizes=[64, 128, 256, 512], post_conv_channel_size=1, bn_eps=None, last_relu=False, bias=False
   ),
   """