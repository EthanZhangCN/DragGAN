import sys
sys.path.append('.')
# sys.path.append('PTI/')
# sys.path.append('PTI/training/')
from random import choice
from string import ascii_uppercase
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
from PTI.configs import global_config, paths_config
import wandb

from PTI.training.coaches.multi_id_coach import MultiIDCoach
from PTI.training.coaches.single_id_coach import SingleIDCoach
from PTI.utils.ImagesDataset import ImagesDataset,Image2Dataset
from PTI.utils.models_utils import load_old_G
import torch

def run_PTI(img,run_name='', use_wandb=False, use_multi_id_training=False):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name

    if use_wandb:
        run = wandb.init(project=paths_config.pti_results_keyword, reinit=True, name=global_config.run_name)
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    embedding_dir_path = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}'
    os.makedirs(embedding_dir_path, exist_ok=True)

    # dataset = ImagesDataset(paths_config.input_data_path, transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))


    G = load_old_G()

    img=img.resize([G.img_resolution,G.img_resolution])
    dataset = Image2Dataset(img)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


    if use_multi_id_training:
        coach = MultiIDCoach(dataloader, use_wandb)
    else:
        coach = SingleIDCoach(dataloader, use_wandb)

    new_G,w_pivot=coach.train()

    return new_G,w_pivot

import pickle
def export_updated_pickle(new_G,out_path):
  print("Exporting large updated pickle based off new generator and ffhq.pkl")
  with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
    d = pickle.load(f)
    old_G = d['G_ema'].cuda() ## tensor
    old_D = d['D'].eval().requires_grad_(False).cpu()

  tmp = {}
  tmp['G_ema'] = old_G.eval().requires_grad_(False).cpu()# copy.deepcopy(new_G).eval().requires_grad_(False).cpu()
  tmp['G'] = new_G.eval().requires_grad_(False).cpu() # copy.deepcopy(new_G).eval().requires_grad_(False).cpu()
  tmp['D'] = old_D
  tmp['training_set_kwargs'] = None
  tmp['augment_pipe'] = None


  with open(out_path, 'wb') as f:
      pickle.dump(tmp, f)

if __name__ == '__main__':
    from PIL import Image
    img = Image.open('PTI/test/test.jpg')
    new_G,w_pivot = run_PTI(img,use_wandb=False, use_multi_id_training=False)
    out_path = f'checkpoints/stylegan2_custom_512_pytorch.pkl'
    export_updated_pickle(new_G,out_path)