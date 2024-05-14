'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import argparse, glob, os, torch, warnings, time
from tools import *
from dataLoader import train_loader
from PiezoBudsModel import PiezoBudsModel
from torch import nn
from torch.utils.mobile_optimizer import optimize_for_mobile

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int,   default=50,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=200,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=20,     help='Batch size (number of users per batch)')
parser.add_argument('--num_uttr', type=int,   default=20,     help='(number of uttrences per user)')
parser.add_argument('--n_cpu',      type=int,   default=4,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=50,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
parser.add_argument('--train_list', type=str,   default="/mnt/ssd/gen/piezo_authentication/train_list_piezo_500ms_1.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser.add_argument('--train_path', type=str,   default="/mnt/hdd/gen/processed_data/wav_clips_500ms/piezobuds_new_1/train/",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
parser.add_argument('--eval_list',  type=str,   default="/mnt/ssd/gen/piezo_authentication/train_list_piezo_500ms_1.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
parser.add_argument('--eval_path',  type=str,   default="/mnt/hdd/gen/processed_data/wav_clips_500ms/piezobuds_new_1/train/",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
parser.add_argument('--eval_user_total', type=int, default=80, help='Number of users for evaluating')
parser.add_argument('--eval_user',  type=int,   default=5)
parser.add_argument('--eval_uttr_enroll',  type=int,   default=30)
parser.add_argument('--eval_uttr_verify',  type=int,   default=5)
parser.add_argument('--eval_noise_type',  type=int,   default=0, help='0: no noise; 1: white noise; 2: conversation; 3: cafe; 4: restaurant; 5: construction')
parser.add_argument('--eval_noise_path',  type=str,   default="/mnt/ssd/gen/GithubRepo/PiezoBuds/noise", help='The path of noise data')
parser.add_argument('--eval_motion_type',  type=int,   default=0, help='0: No motion; 1: turn; 2: tap; 3: clap; 4: walk')
parser.add_argument('--eval_motion_path',  type=str,   default="/mnt/ssd/gen/GithubRepo/PiezoBuds/motion", help='The path of motion data')
parser.add_argument('--musan_path', type=str,   default="/mnt/hdd/gen/musan/musan",                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
parser.add_argument('--rir_path',   type=str,   default="/mnt/hdd/gen/rirs_noises/RIRS_NOISES/simulated_rirs",     help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case');
parser.add_argument('--save_path',  type=str,   default="exps/huber3",                                     help='Path to save the score.txt and models')
parser.add_argument('--initial_model',  type=str,   default="exps/ge2e_apzff_noncurrent_04051101_full_dataset/model/model_1900.model",          help='Path of the initial_model')
# parser.add_argument('--export_mobile_model', type=bool, default=False)
parser.add_argument('--initial_extractor', type=str, default="/mnt/ssd/gen/GithubRepo/PiezoBuds/ECAPA-TDNN-main/exps/ours/model/model_0080.model", help="")

## Model and Loss settings
parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int,   default=81,   help='Number of speakers')

parser.add_argument('--device',  type=str,   default=device)

## Command
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)

## Define the data loader
trainloader = train_loader(**vars(args))
_, veri_list = trainloader.return_user_lists()
trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)

## Search for the exist models
modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
modelfiles.sort()

## Only do evaluation, the initial_model is necessary
# if args.eval == True:
if True:
	s = PiezoBudsModel(**vars(args))
	print("Model %s loaded from previous state!"%args.initial_model)
	s.load_parameters(args.initial_model)
	s.eval()

	# example_e = torch.rand(1, 8000)
	# example_c = torch.rand(2, 3, 8, 8)
	# traced_script_extractor_a = torch.jit.trace(s.encoder_a, (example_e))
	# traced_script_extractor_a = optimize_for_mobile(traced_script_extractor_a)
	# traced_script_extractor_a._save_for_lite_interpreter('./mobile_model/' + 'extractor_a_m.ptl')

	# traced_script_extractor_p = torch.jit.trace(s.encoder_p, (example_e))
	# traced_script_extractor_p = optimize_for_mobile(traced_script_extractor_p)
	# traced_script_extractor_p._save_for_lite_interpreter('./mobile_model/' + 'extractor_p_m.ptl')

	# traced_script_converter = torch.jit.trace(s.converter, example_c)
	# traced_script_converter = optimize_for_mobile(traced_script_converter)
	# traced_script_converter._save_for_lite_interpreter('./mobile_model/' + 'converter_m.ptl')

	# if torch.cuda.device_count() > 1:
	# 	print("Using", torch.cuda.device_count(), "GPUs!")
	# 	s = nn.DataParallel(s)

	EER, minDCF, thres, FAR_replay, EER_audio, EER_piezo, thres_audio, thres_piezo, thres_conv = s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path, eval_user_total=args.eval_user_total,
							eval_user=args.eval_user, eval_uttr_enroll=args.eval_uttr_enroll, 
							eval_uttr_verify=args.eval_uttr_verify, veri_usr_lst=veri_list, 
							eval_noise_type=args.eval_noise_type, eval_noise_path=args.eval_noise_path,
							eval_motion_type=args.eval_motion_type, eval_motion_path=args.eval_motion_path,
							eval_times=2000)
	print("EER %2.2f%%, threshold %2.5f, minDCF %.4f%%, FAR_replay %.4f, EER_audio %2.2f%%, EER_piezo %2.2f%%, thres_audio %2.5f, thres_piezo %2.5f, thres_conv %2.5f"%(EER * 100, 
																																									 thres, minDCF, FAR_replay, EER_audio * 100, EER_piezo * 100, thres_audio, thres_piezo, thres_conv))
	quit()

## If initial_model is exist, system will train from the initial_model
if args.initial_model != "":
	print("Model %s loaded from previous state!"%args.initial_model)
	s = PiezoBudsModel(**vars(args))
	s.load_parameters(args.initial_model)
	epoch = 1

## Otherwise, system will try to start from the saved model&epoch
elif len(modelfiles) >= 1:
	print("Model %s loaded from previous state!"%modelfiles[-1])
	epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
	s = PiezoBudsModel(**vars(args))
	s.load_parameters(modelfiles[-1])
## Otherwise, system will train from scratch
else:
	epoch = 1
	s = PiezoBudsModel(**vars(args))
	if args.initial_extractor != "":
		loaded_state = torch.load(args.initial_extractor, map_location=device)
		state_a = s.encoder_a.state_dict()
		state_p = s.encoder_p.state_dict()
		for name, param in loaded_state.items():
			origname = name
			name = remove_prefix(origname, 'speaker_encoder.')
			if name in state_a:
				if state_a[name].size() == loaded_state[origname].size():
					state_a[name].copy_(loaded_state[origname])
					state_p[name].copy_(loaded_state[origname])
		s.encoder_a.load_state_dict(state_a)
		s.encoder_p.load_state_dict(state_p)


EERs = []
score_file = open(args.score_save_path, "a+")

while(1):
	## Training for one epoch
	loss, lr, acc = s.train_network(epoch = epoch, loader = trainLoader)

	## Evaluation every [test_step] epochs
	if epoch % args.test_step == 0:
		s.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)
		EERs.append(s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path, 
							 eval_user=args.eval_user, eval_uttr_enroll=args.eval_uttr_enroll, 
							 eval_uttr_verify=args.eval_uttr_verify, veri_usr_lst=veri_list)[0])
		print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%"%(epoch, acc, EERs[-1] * 100, min(EERs) * 100))
		score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n"%(epoch, lr, loss, acc, EERs[-1] * 100, min(EERs) * 100))
		score_file.flush()

	if epoch >= args.max_epoch:
		quit()

	epoch += 1
