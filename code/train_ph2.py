import copy
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1, 2, 3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1, 2, 3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1" # "0, 1, 2, 3"

import timm
from datasets.dataset import NPY_datasets, Polyp_datasets, Isic_datasets
from tensorboardX import SummaryWriter

from engine import *

import sys
import torch
from torch.utils.data import DataLoader
from utils import *


from configs.ph2.config_setting_atrous import *
from configs.ph2.config_setting_other_network import *

from models.Atrous.atrous_UL_CNN import atrous_ULPSR_basev3_CNN, atrous_ULP_basev3_CNN, atrous_ULPSR_basev3_CNN_stru2, atrous_ULPSR_basev3_CNN_stru18     # our model
from models.Atrous.atrous_vmunet import atrous_VMUNet

from models.other_network.C2SDG.unet_ccsdg import UNetCCSDG
from models.other_network.egenet.egeunet import EGEUNet
from models.other_network.HMT_UNet.hmt_unet import HMTUNet
from models.other_network.Hvmunet.H_vmunet import H_vmunet
from models.other_network.malunet.malunet import MALUNet
from models.other_network.UltraLight.UltraLight_VM_UNet import UltraLight_VM_UNet
from models.other_network.unet_v2.UNet_v2 import UNetV2
from models.other_network.vmunet_v1.vmunet import VMUNet
from models.other_network.vmunet_v2.vmunet_v2 import VMUNetV2

import warnings

warnings.filterwarnings("ignore")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print(config.work_dir)

    print('#----------GPU init----------#')
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    # train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_dataset = Isic_datasets(config.data_path, config, train=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)

    # val_dataset = NPY_datasets(config.data_path, config, train=False)

    val_loader_dict = {}

    val_dataset = Isic_datasets(config.data_path, config, train=False, test_dataset='isic16')
    val_loader = DataLoader(val_dataset,
                            batch_size=config.val_batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True
                            )

    val_loader_dict['isic16'] = val_loader

    print('#----------Prepareing Model----------#')
    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    # Atrous_UL_CNN
    if config.network == 'atrous_UL_CNN':
        print('this is for atrous UL CNN')
        model=atrous_ULPSR_basev3_CNN(
            num_classes=model_cfg['num_classes'], input_channels=model_cfg['input_channels'],
            c_list=model_cfg['c_list'], d_conv=model_cfg['d_conv'],
            split_att=model_cfg['split_att'], bridge=model_cfg['bridge'],
            if_shifted_round=model_cfg['if_shifted_round'], if_ss2d=model_cfg['if_ss2d'],
            forward_type=model_cfg['forward_type'],
            encoder_atrous_step=model_cfg['encoder_atrous_step'], decoder_atrous_step=model_cfg['decoder_atrous_step'],
            if_CNN=model_cfg['if_CNN'], if_SE=model_cfg['if_SE'], if_SK=model_cfg['if_SK'],
        )
    elif config.network == 'atrous_UL_CNN_noshift':
        print('this is for atrous UL CNN no shift')
        model = atrous_ULP_basev3_CNN(
            num_classes=model_cfg['num_classes'], input_channels=model_cfg['input_channels'],
            c_list=model_cfg['c_list'], d_conv=model_cfg['d_conv'],
            split_att=model_cfg['split_att'], bridge=model_cfg['bridge'],
            if_ss2d=model_cfg['if_ss2d'],
            forward_type=model_cfg['forward_type'],
            encoder_atrous_step=model_cfg['encoder_atrous_step'], decoder_atrous_step=model_cfg['decoder_atrous_step'],
            if_CNN=model_cfg['if_CNN'], if_SE=model_cfg['if_SE'], if_SK=model_cfg['if_SK'],
        )
    elif config.network == 'atrous_UL_CNN_stru2':
        print('this is for atrous UL CNN structure [2,2,2,2,2,2]')
        model=atrous_ULPSR_basev3_CNN_stru2(
            num_classes=model_cfg['num_classes'], input_channels=model_cfg['input_channels'],
            c_list=model_cfg['c_list'], d_conv=model_cfg['d_conv'],
            split_att=model_cfg['split_att'], bridge=model_cfg['bridge'],
            if_shifted_round=model_cfg['if_shifted_round'], if_ss2d=model_cfg['if_ss2d'],
            forward_type=model_cfg['forward_type'],
            encoder_atrous_step=model_cfg['encoder_atrous_step'], decoder_atrous_step=model_cfg['decoder_atrous_step'],
            if_CNN=model_cfg['if_CNN'], if_SE=model_cfg['if_SE'], if_SK=model_cfg['if_SK'],
        )
    elif config.network == 'atrous_UL_CNN_stru18':
        print('this is for atrous UL CNN structure [2,2,2,2,16,2]')
        model = atrous_ULPSR_basev3_CNN_stru18(
            num_classes=model_cfg['num_classes'], input_channels=model_cfg['input_channels'],
            c_list=model_cfg['c_list'], d_conv=model_cfg['d_conv'],
            split_att=model_cfg['split_att'], bridge=model_cfg['bridge'],
            if_shifted_round=model_cfg['if_shifted_round'], if_ss2d=model_cfg['if_ss2d'],
            forward_type=model_cfg['forward_type'],
            encoder_atrous_step=model_cfg['encoder_atrous_step'], decoder_atrous_step=model_cfg['decoder_atrous_step'],
            if_CNN=model_cfg['if_CNN'], if_SE=model_cfg['if_SE'], if_SK=model_cfg['if_SK'],
        )
    # other network
    elif config.network == 'C2SDG':
        print('this is for C2SDG')
        model = UNetCCSDG()
    elif config.network == 'egeunet':
        print('this is for egeunet')
        model = EGEUNet()
    elif config.network == 'HMT_UNet':
        print('this is for HMT_UNet')
        model = HMTUNet()
    elif config.network == 'H_vmunet':
        print(' this is for H_vmunet')
        model = H_vmunet()
    elif config.network == 'malunet':
        print('this is for malunet')
        model = MALUNet()
    elif config.network == 'UltraLight':
        print("this is for UltraLight_VM_UNet")
        model = UltraLight_VM_UNet()        # default value
    elif config.network == 'unetv2':
        print('this is for UNet-v2')
        model = UNetV2()
    elif config.network == 'vmunet_v1':
        print("this is VMUNet_V1")
        model = VMUNet()
    elif config.network == 'vmunet_v2':
        print("this is VMUNet_V2")
        model = VMUNetV2(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
            deep_supervision=model_cfg['deep_supervision'],
        )
        model.load_from()
    else:
        raise Exception('network in not right!')

    model = model.to(device=device)
    # model = torch.nn.DataParallel(model.cuda(), device_ids=config.gpu_id, output_device=config.gpu_id[0])

    cal_params_flops(copy.deepcopy(model), 256, logger)
    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    max_dsc = -100
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cuda'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss, max_dsc = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss'], checkpoint['max_dsc']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        t = time.time()
        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )
        print(f'training time: {time.time()-t}s')
        loss_all = []

        for name in ['isic16']:
            val_loader_t = val_loader_dict[name]

            loss_t, dsc = val_one_epoch(
                val_loader_t,
                model,
                criterion,
                epoch,
                logger,
                config,
                val_data_name=name
            )
            loss_all.append(loss_t)
        print(f'total time: {time.time()-t}s')
        loss = np.mean(loss_all)

        # if loss < min_loss:
        if dsc > max_dsc:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch
            max_dsc = dsc
        print(f'the best.pth {min_epoch} and best dsc is {max_dsc}')

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'max_dsc': max_dsc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

        # 测试的步骤：也要按照 Polyp 的文件夹列表
    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cuda'))
        model.load_state_dict(best_weight)

        # val_dataset = Isic_datasets(config.data_path, config, train=False, test_dataset=dataset)
        # val_dataset = val_loader_dict['isic17']

        test_loader = DataLoader(val_dataset,
                                batch_size=config.test_batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                drop_last=False
                                )


        for name in ['isic16']:
            # val_loader_t = val_loader_dict[name]
            loss, log_info = test_one_epoch(
                # val_loader_t,
                test_loader,
                model,
                criterion,
                logger,
                config,
                test_data_name=name
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )

    return log_info + " " + config.work_dir


if __name__ == '__main__':

    """ out model """
    # config = setting_config_atrousv2_ULPSR_step2_CNN_SE_SK            # OVER4
    # config = setting_config_atrousv2_ULPSR_step2_CNN_SE_SK_small      # OVER
    # config = setting_config_atrousv2_ULPSR_step2_CNN_SE_SK_big        # OVER4

    """vallian"""
    # config = setting_config_vallian_ULPSR_CNN_SE_SK                   # OVER4
    # config = setting_config_vallian_ULPSR_CNN_SE_SK_small             # OVER2

    """ efficient scan"""
    # config = setting_config_efficient_ULPSR_step2_CNN_SE_SK           # OVER2
    # config = setting_config_efficient_ULPSR_step2_CNN_SE_SK_small

    """ atrous ss2d scan"""
    # config = setting_config_atrousv2_ULPSR_ss2d_CNN_SE_SK             # OVER4
    # config = setting_config_atrousv2_ULPSR_ss2d_CNN_SE_SK_small       # OVER

    """ ss2d """
    # config = setting_config_ULPSR_ss2d_CNN_SE_SK          # OVER
    # config = setting_config_ULPSR_ss2d_CNN_SE_SK_small    # OVER

    """structure ablation study"""
    # config = setting_config_atrousv2_ULPSR_step2_CNN_SE_SK_small_stru2
    # config = setting_config_atrousv2_ULPSR_step2_CNN_SE_SK_small_stru18

    """ component ablation study"""
    # config = setting_config_ULP_small                         # OVER6
    # config = setting_config_ULPS_small                        # OVER4
    # config = setting_config_ULPSR_small                       # OVER4
    # config = setting_config_atrousv2_ULPSR_small              # OVER4
    # config = setting_config_atrousv2_ULPSR_CNN_small          # OVER4
    # config = setting_config_atrousv2_ULPSR_CNN_SE_small       # OVER4
    # config = setting_config_atrousv2_ULPSR_CNN_SK_small       # OVER4

    """ atrous step ablation study"""
    # config = setting_config_atrousv2_ULPSR_step3_CNN_SE_SK_small            # OVER4
    # config = setting_config_atrousv2_ULPSR_step4_CNN_SE_SK_small            # OVER4
    # config = setting_config_atrousv2_ULPSR_step8_CNN_SE_SK_small            # OVER2
    # config = setting_config_atrousv2_ULPSR_step23_CNN_SE_SK_small           # OVER4
    # config = setting_config_atrousv2_ULPSR_step12_CNN_SE_SK_small           # OVER4
    # config = setting_config_atrousv2_ULPSR_step222221_CNN_SE_SK_small       # OVER4
    config = setting_config_atrousv2_ULPSR_step222224_CNN_SE_SK_small

    """ other network """
    # config = setting_config_C2SDG           # OVER4
    # config = setting_config_egeunet         # OVER4
    # config = setting_config_HMTUnet         # OVER4
    # config = setting_config_Hvmunet         # OVER4
    # config = setting_config_malunet         # OVER4
    # config = setting_config_UL              # OVER4
    # config = setting_config_Unetv2
    # config = setting_config_vmunet_v1       # OVER4
    # config = setting_config_VMUNetv2        # OVER4


    """ ablation study of stride"""
    # config = setting_config_atrousv2_ULPSR_step1_CNN_SE_SK      # stride is 1
    # config = setting_config_atrousv2_ULPSR_step3_CNN_SE_SK      # stride is 3
    # config = setting_config_atrousv2_ULPSR_step4_CNN_SE_SK      # stride is 4
    # config = setting_config_atrousv2_ULPSR_step5_CNN_SE_SK      # stride is 5
    # config = setting_config_atrousv2_ULPSR_step6_CNN_SE_SK      # stride is 6
    # config = setting_config_atrousv2_ULPSR_step7_CNN_SE_SK      # stride is 7
    # config = setting_config_atrousv2_ULPSR_step8_CNN_SE_SK      # stride is 8

    # config = setting_config_atrousv2_ULPSR_step23_CNN_SE_SK           # stride is 2, then 3
    # config = setting_config_atrousv2_ULPSR_step12_CNN_SE_SK           # stride is 1, then 2
    # config = setting_config_atrousv2_ULPSR_stepS12_CNN_SE_SK          # first 3 stage is 1, later 2 stage is 2
    # config = setting_config_atrousv2_ULPSR_stepS21_CNN_SE_SK          # first 3 stage is 2, later 2 stage is 1
    # config = setting_config_atrousv2_ULPSR_stepS54321_CNN_SE_SK       # stage 54321
    # config = setting_config_atrousv2_ULPSR_stepS12345_CNN_SE_SK       #
    # config = setting_config_atrousv2_ULPSR_stepS222224_CNN_SE_SK      #
    # config = setting_config_atrousv2_ULPSR_stepS222221_CNN_SE_SK      #
    # config = setting_config_atrousv2_ULPSR_stepS84221_CNN_SE_SK       #


    log_info1 = main(config)
    # log_info2 = main(config)
    print(log_info1)
    # print()
    # print(log_info2)
