import torch
import torch.backends.cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from model.utils import Reconstruction3DDataLoader
from model.reconstruction_model import *
from utils import *
import glob
import os
import argparse
import time
from tqdm import tqdm
import my_filters


parser = argparse.ArgumentParser(description="LUSS AE Testing")
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--t', type=int, default=17, help='height of input images')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='shanghai', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='dataset', help='directory of data')
parser.add_argument('--model_path', type=str, help='path to learned model (.pth)')
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--np_save_dir', type=str, default='npy/', help='numpy save dir')
parser.add_argument('--ckpt_step', type=int, default=1, help='ckpt step')
parser.add_argument('--save_npy', type=int, default=1, help='ckpt step')

args = parser.parse_args()
np_dir = args.np_save_dir

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus[0]

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

## Calling model components
encoder_model = Reconstruction3DEncoder(chnum_in=1)
decoder_model = Decoder_recon_pred(chnum_in=1)
feature_projector = projection_head()
embed_1 = embedding_1()
embed_2 = embedding_2()
irreg_predictor = class_predictor()

## Enabling data parallelisation for mutli-gpu support
encoder_model = nn.DataParallel(encoder_model)
decoder_model = nn.DataParallel(decoder_model)
feature_projector = nn.DataParallel(feature_projector)
embed_1 = nn.DataParallel(embed_1)
embed_2 = nn.DataParallel(embed_2)
irreg_predictor = nn.DataParallel(irreg_predictor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load learned model components and map to chosen device
model_dict = torch.load(args.model_path, map_location=device)
encoder_model.load_state_dict(model_dict['encoder_model'].state_dict())
decoder_model.load_state_dict(model_dict['decoder_model'].state_dict())
feature_projector.load_state_dict(model_dict['feature_projector'].state_dict())
embed_1.load_state_dict(model_dict['embed_1'].state_dict())
embed_2.load_state_dict(model_dict['embed_2'].state_dict())
irreg_predictor.load_state_dict(model_dict['irreg_predictor'].state_dict())

# Enable cuda support
encoder_model.cuda()
decoder_model.cuda()
feature_projector.cuda()
embed_1.cuda()
embed_2.cuda()
irreg_predictor.cuda()

# Enable evaluation mode
encoder_model.eval()
decoder_model.eval()
irreg_predictor.eval()
feature_projector.eval()
embed_1.eval()
embed_2.eval()

labels = np.load('./data/frame_labels_' + args.dataset_type + '.npy')

# Loading dataset
test_folder = os.path.join(args.dataset_path, args.dataset_type, 'testing', 'frames')
img_extension = '.tif' if args.dataset_type == 'ped1' else '.jpg'
test_dataset = Reconstruction3DDataLoader(test_folder, transforms.Compose([transforms.ToTensor(), ]),
                                          resize_height=args.h, resize_width=args.w, dataset=args.dataset_type,
                                          img_extension=img_extension, num_frames=args.t)
test_size = len(test_dataset)
test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                             num_workers=args.num_workers_test, drop_last=False)

videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*/')))
for video in videos_list:
    video_name = video.split('/')[-2]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + img_extension))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])

labels_list = []
label_length = 0
a_vcr = []
a_ffp = []
n_prp = []

print('Evaluation of ', args.dataset_type)

# Setting for video anomaly detection
for video in sorted(videos_list):
    video_name = video.split('/')[-2]
    labels_list = np.append(labels_list,
                            labels[0][16 + label_length:videos[video_name]['length'] + label_length])
    label_length += videos[video_name]['length']

tic = time.time()
the_filter = my_filters.GaussPool2D((5, 5), sigma=(1,1), stride=2)
pbar = tqdm(total=len(test_batch))
for ki, (imgs) in enumerate(test_batch):
    imgs = Variable(imgs).cuda()
    with torch.no_grad():
        encoding = encoder_model(imgs[:, :, :-1, :, :])
        p_output, outputs = decoder_model(encoding)
        res_recon = outputs - imgs[:, :, :-1, :, :]
        res_pred = p_output - imgs[:, :, -1, :, :].unsqueeze(2)

        a_vcr += bpe_clip(res_recon[:, 0, :, :, :], the_filter)
        a_ffp += bpe_clip(res_pred[:, 0, :, :, :], the_filter)

        prp_outputs = F.softmax(irreg_predictor(embed_2(embed_1(feature_projector(encoding)))), dim=1)
        n_prp += prp_outputs[:, 0].tolist()  # 1 for irreg, this is for reg, # high score is normal

    pbar.set_postfix({
        'Testing': 'iter {:d}/{:d}'.format(ki, len(test_batch))
    })

    pbar.update(1)

pbar.close()
if (args.save_npy == 1):
    print("Saving frame level anomaly scores of each task, without rescaling..")
    if not os.path.isdir(np_dir):
        os.mkdir(np_dir)
    np.save("{}/vid_list.npy".format(np_dir), videos_list)
    np.save("{}/a_vcr.npy".format(np_dir), a_vcr)
    np.save("{}/a_ffp.npy".format(np_dir), a_ffp)
    np.save("{}/n_prp.npy".format(np_dir), n_prp)
    np.save("{}/labels.npy".format(np_dir), labels_list)

alphas = [0.2, 0.2, 0.9] # this for shanghai, set accordingly as defined in paper
anomaly_score_list = []
vid_start_index = 0
for vi, video in enumerate(sorted(videos_list)):
    vid_end_index = len(glob.glob(os.path.join(video, '*' + '.jpg'))) - 16
    a_vcr_vid = a_vcr[vid_start_index: vid_start_index + vid_end_index]
    a_ffp_vid = a_ffp[vid_start_index: vid_start_index + vid_end_index]
    n_prp_vid = n_prp[vid_start_index: vid_start_index + vid_end_index]
    a_prp_vid = list(1 - np.array(n_prp_vid))
    a_vcr_vid = rescaled_score_per_video(a_vcr_vid, "robust")
    a_ffp_vid = rescaled_score_per_video(a_ffp_vid, "robust")
    a_prp_vid = rescaled_score_per_video(a_prp_vid, "min_max")
    anomaly_score_list += cumulative_score(a_vcr_vid, a_ffp_vid, a_prp_vid, alphas[0], alphas[1], alphas[2])
    vid_start_index += vid_end_index
anomaly_score_list = np.asarray(anomaly_score_list)
micro_auc = AUC(anomaly_score_list, np.expand_dims(labels_list, 0))
print('AUC: {:.4f}%'.format(micro_auc))