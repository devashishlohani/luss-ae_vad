import torch.backends.cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from model.utils import Reconstruction3DDataLoader, Reconstruction3DDataLoaderJump
from model.reconstruction_model import *
from utils import *
from tqdm import tqdm
import argparse
import sys

parser = argparse.ArgumentParser(description="LUSS AE Training")
parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--t', type=int, default=17, help='number of frames')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate phase 1')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for the train loader')
parser.add_argument('--dataset_type', type=str, default='ped2', choices=['ped2', 'avenue', 'shanghai'],
                    help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='dataset', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='basename of folder to save weights')
parser.add_argument('--loss_recon', type=float, default=0.6, help='weight of the reconstruction loss')
parser.add_argument('--loss_pred', type=float, default=0.4, help='weight of the frame prediction loss')
parser.add_argument('--loss_irr', type=float, default=1, help='weight of the PRP loss')
parser.add_argument('--model_dir', type=str, default=None, help='path of model for resume')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch. usually number in filename + 1')
parser.add_argument('--prob_accelerated_clip', type=float, default=0.5, help='probability of acclerated clip')
parser.add_argument('--jump', nargs='+', type=int, default=[2, 3, 4, 5], help='skip frames number (hyperparameter s)')
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus[0]

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

exp_dir = args.exp_dir
exp_dir += 'lr' + str(args.lr) if args.lr != 1e-4 else ''
exp_dir += '_jump_prob_' + str(args.prob_accelerated_clip) if args.prob_accelerated_clip != 0 else ''
exp_dir += '_jump[' + ','.join(
    [str(args.jump[i]) for i in range(0, len(args.jump))]) + ']' if args.prob_accelerated_clip != 0 else ''

print('exp_dir: ', exp_dir)

train_folder = os.path.join(args.dataset_path, args.dataset_type, 'training', 'frames')

# Loading dataset
img_extension = '.tif' if args.dataset_type == 'ped1' else '.jpg'
train_dataset = Reconstruction3DDataLoader(train_folder, transforms.Compose([transforms.ToTensor()]),
                                           resize_height=args.h, resize_width=args.w, dataset=args.dataset_type,
                                           img_extension=img_extension, num_frames=args.t)
train_dataset_jump = Reconstruction3DDataLoaderJump(train_folder, transforms.Compose([transforms.ToTensor()]),
                                                    resize_height=args.h, resize_width=args.w,
                                                    dataset=args.dataset_type, jump=args.jump,
                                                    return_normal_seq=args.prob_accelerated_clip > 0,
                                                    img_extension=img_extension)

train_size = len(train_dataset)

train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
train_batch_jump = data.DataLoader(train_dataset_jump, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, drop_last=True)

# Report the training process
log_dir = os.path.join('./exp', args.dataset_type, exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'), 'a')
sys.stdout = f

torch.set_printoptions(profile="full")

loss_func_mse = nn.MSELoss(reduction='none')
loss_func_ce = nn.CrossEntropyLoss()

if args.start_epoch < args.epochs:
    encoder_model = Reconstruction3DEncoder(chnum_in=1)
    decoder_model = Decoder_recon_pred(chnum_in=1)
    feature_projector = projection_head()
    embed_1 = embedding_1()
    embed_2 = embedding_2()
    irreg_predictor = class_predictor()

    encoder_model = nn.DataParallel(encoder_model)
    decoder_model = nn.DataParallel(decoder_model)
    feature_projector = nn.DataParallel(feature_projector)
    embed_1 = nn.DataParallel(embed_1)
    embed_2 = nn.DataParallel(embed_2)
    irreg_predictor = nn.DataParallel(irreg_predictor)

    encoder_model.cuda()
    decoder_model.cuda()
    feature_projector.cuda()
    embed_1.cuda()
    embed_2.cuda()
    irreg_predictor.cuda()

    params = list(encoder_model.parameters()) +\
             list(decoder_model.parameters()) + \
             list(feature_projector.parameters()) + \
             list(embed_1.parameters()) + \
             list(embed_2.parameters()) + \
             list(irreg_predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # resume
    if args.model_dir is not None:
        assert args.start_epoch > 0
        # Loading the trained model
        model_dict = torch.load(args.model_dir)
        encoder_model.load_state_dict(model_dict['encoder_model'].state_dict())
        decoder_model.load_state_dict(model_dict['decoder_model'].state_dict())
        feature_projector.load_state_dict(model_dict['feature_projector'].state_dict())
        embed_1.load_state_dict(model_dict['embed_1'].state_dict())
        embed_2.load_state_dict(model_dict['embed_2'].state_dict())
        irreg_predictor.load_state_dict(model_dict['irreg_predictor'].state_dict())
        optimizer.load_state_dict(model_dict['optimizer'])

    for epoch in range(args.start_epoch, args.epochs):
        pbar = tqdm(total=len(train_batch))
        lossepoch = 0
        lossepochpred = 0
        losscounter = 0
        l_prp = 0
        l_tot = 0

        for j, (imgs, imgsjump) in enumerate(zip(train_batch, train_batch_jump)):
            reg_batch = copy.deepcopy(imgsjump[1])  # regular or skip-frame batch
            reg_batch = reg_batch.cuda()
            irr_label = [] # 1 means irregular

            for b in range(args.batch_size):
                total_pseudo_prob = 0
                prob_accelerated_clip = total_pseudo_prob <= np.random.rand() < total_pseudo_prob + args.prob_accelerated_clip
                if prob_accelerated_clip:
                    reg_batch[b] = imgsjump[0][b].cuda()    #skipped window
                    irr_label.append(1)
                else:
                    irr_label.append(0)

            ########## recon + prediction
            normal_batch = Variable(imgs).cuda()
            p_output, outputs = decoder_model.forward(encoder_model.forward(normal_batch[:, :, :-1, :, :]))  # (4,1,16,256,256)
            loss_mse = loss_func_mse(outputs, normal_batch[:, :, :-1, :, :])  # reconstruction loss
            loss_mse_pred = loss_func_mse(p_output, normal_batch[:, :, -1, :, :].unsqueeze(2))  # (4,1,256,256) # prediction loss

            reg_features = feature_projector.forward(encoder_model.forward(reg_batch[:, :, :-1, :, :])) #512 features
            ########## irreg prediction for PRP task
            p_irreg = irreg_predictor.forward(embed_2.forward(embed_1.forward(reg_features))) #2
            loss_p_irr = loss_func_ce(p_irreg, torch.LongTensor(irr_label).cuda())

            modified_loss_mse = []
            modified_loss_mse_pred = []
            for b in range(args.batch_size):
                modified_loss_mse.append(torch.mean(loss_mse[b]))
                modified_loss_mse_pred.append(torch.mean(loss_mse_pred[b]))
                lossepoch += modified_loss_mse[-1].cpu().detach().item()
                lossepochpred += modified_loss_mse_pred[-1].cpu().detach().item()
                losscounter += 1

            assert len(modified_loss_mse) == loss_mse.size(0)
            assert len(modified_loss_mse_pred) == loss_mse_pred.size(0)
            stacked_loss_mse = torch.stack(modified_loss_mse)
            loss_recon = torch.mean(stacked_loss_mse)
            stacked_loss_mse_pred = torch.stack(modified_loss_mse_pred)
            loss_pred = torch.mean(stacked_loss_mse_pred)

            loss = args.loss_pred * loss_pred + args.loss_recon * loss_recon + args.loss_irr * loss_p_irr
            l_prp += args.loss_irr * loss_p_irr.cpu().detach().item()
            l_tot += loss.cpu().detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({
                'Epoch': '{:d} iter {:d}/{:d}'.format(epoch, j, len(train_batch)),
                'total_loss': '{:.6f}'.format(loss.item())})
            pbar.update(1)

        l_r = lossepoch/losscounter
        l_p = lossepochpred/losscounter
        l_prp = l_prp/j
        l_tot = l_tot/j
        print('----------------------------------------')
        print('Epoch:', epoch)
        if losscounter != 0:
            print('MeanLoss: Reconstruction {:.9f}'.format(l_r))
            print('MeanLoss: Pred {:.9f}'.format(l_p))
            print('MeanLoss: PRP_loss {:.9f}'.format(l_prp))
            print('MeanLoss: Total_loss {:.9f}'.format(l_tot))
        pbar.close()

        if(epoch%10 == 0):
            # Save the model
            model_dict = {
                'encoder_model': encoder_model,
                'decoder_model': decoder_model,
                'irreg_predictor': irreg_predictor,
                'feature_projector': feature_projector,
                'embed_1': embed_1,
                'embed_2': embed_2,
                'optimizer': optimizer.state_dict()
            }
            torch.save(model_dict, os.path.join(log_dir, 'model_{:02d}.pth'.format(epoch)))

print('Training is finished')
sys.stdout = orig_stdout
f.close()




