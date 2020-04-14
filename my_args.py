import os
import datetime
import argparse
import numpy
import networks
import  torch

modelnames =  networks.__all__
# import datasets
datasetNames = ('Vimeo_90K_interp') #datasets.__all__

class Args:
  debug =  ** action 'store_true'; help='Enable debug mode'
  netName = 'DAIN'
  datasetName = 'Vimeo_90K_interp'
  datasetPath = ''
  dataset_split = 97
  seed = 1
  numEpoch = 100
  batch_size = 1
  workers = 8
  channels = 3
  filter_size = 4
  lr = 0.002
  rectify_lr = 0.001
  save_which = 1
  time_step = 0.5
  flow_lr_coe = 0.01
  occ_lr_coe = 1.0
  filter_lr_coe = 1.0
  ctx_lr_coe = 1.0
  depth_lr_coe = 0.001
  deblur_lr_coe = 0.01
  alpha = [0.0, 1.0]
  epsilon = 1e-6
  weight_decay = 0
  patience = 5
  factor = 0.2
  SAVED_MODEL = None
  no_date =  ** action 'store_true'; help='don\'
  use_cuda = True
  use_cudnn = 1
  dtype = torch.cuda.FloatTensor
  resume = ''
  uid = None
  force =  ** action 'store_true'; help='force to override the given uid'
  save_path = save_path
  log = save_path+'/log.txt'
  arg = save_path+'/args.txt'

args=Args()
