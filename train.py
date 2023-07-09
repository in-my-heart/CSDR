import torch.optim as optim
import glob
import json
import argparse
import os
import random
from time import gmtime, strftime
from model import *
from dataset import FeatDataLayer, DATA_LOADER
from utils import *
from sklearn.metrics.pairwise import cosine_similarity
import torch.backends.cudnn as cudnn
import classifier
from kmeans_pytorch import kmeans
from loss import SupConLoss
from fast_pytorch_kmeans import KMeans
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='dataset: CUB, AWA1, FLO, SUN, APY')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--image_embedding', default='res101', type=str)
parser.add_argument('--class_embedding', default='att', type=str)
parser.add_argument('--finetune', type=bool, default=True, help='Use fine-tuned feature')

parser.add_argument('--gen_nepoch', type=int, default=400, help='number of epochs to train for')

parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train gen')  # 0.0001

parser.add_argument('--vae', type=float, default=1, help='vae weight')
parser.add_argument('--align', type=float, default=1, help='align weight')
parser.add_argument('--rec', type=float, default=1, help='rec weight')
parser.add_argument('--swap', type=float, default=1, help='swap weight')
parser.add_argument('--bet', type=float, default=1, help='between weight')
parser.add_argument('--intra', type=float, default=1, help='intra set weight')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight_decay')
parser.add_argument('--dis', type=float, default=0.3, help='')

parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--batchsize', type=int, default=512, help='input batch size')
parser.add_argument('--nSample', type=int, default=1200, help='number features to generate per class')

parser.add_argument('--disp_interval', type=int, default=50)
parser.add_argument('--save_interval', type=int, default=10000)
parser.add_argument('--evl_interval', type=int, default=50)
parser.add_argument('--manualSeed', type=int, default=110, help='manual seed')

parser.add_argument('--latent_dim', type=int, default=128, help='dimention of latent z')
parser.add_argument('--gh_dim', type=int, default=4096, help='dimention of hidden layer in decoder')
parser.add_argument('--eh_dim', type=int, default=4096, help='dimention of hidden layer in encoder')
parser.add_argument('--q_z_nn_output_dim', type=int, default=128, help='dimention of latent z')
parser.add_argument('--twice', type=bool, default=True, help='Use both vae\'s output and original input in ae')
parser.add_argument('--CS_dim', type=int, default=1024, help='dimention of class-shared')
parser.add_argument('--CU_dim', type=int, default=1024, help='dimention of class-unique')
parser.add_argument('--UNS_dim', type=int, default=1024, help='dimention of semantic-unspecific')
parser.add_argument('--centernum', type=int, default=10)
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
opt = parser.parse_args()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ': ')))
opt.gpu = torch.device("cuda:" + opt.gpu if torch.cuda.is_available() else "cpu")


def train():
    dataset = DATA_LOADER(opt)
    opt.C_dim = dataset.att_dim
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    opt.y_dim = dataset.ntrain_class
    out_dir = 'out/{}/lr-{}_ds-{}__nS-{}_nZ-{}_bs-{}_gh-{}_eh-{}'.format(opt.dataset,
                                                                         opt.lr,
                                                                         opt.CS_dim, opt.nSample, opt.Z_dim,
                                                                         opt.batchsize, opt.gh_dim,
                                                                         opt.eh_dim)
    os.makedirs(out_dir, exist_ok=True)
    print("The output dictionary is {}".format(out_dir))

    log_dir = out_dir + '/log_{}.txt'.format(opt.dataset)
    with open(log_dir, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    dataset.feature_dim = dataset.train_feature.shape[1]
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    opt.y_dim = dataset.ntrain_class

    data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.cpu().numpy(), opt)

    opt.niter = int(dataset.ntrain / opt.batchsize) * opt.gen_nepoch

    result_gzsl_soft = Result()
    result_zsl_soft = Result()
    vae = VAE(opt).to(opt.gpu)
    alignNet = AlignNet(opt).to(opt.gpu)
    ae = AE(opt).to(opt.gpu)

    with open(log_dir, 'a') as f:
        f.write('\n')
        f.write('Generative Model Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step = 0
    optimizer = optim.Adam(vae.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    alignNet_optimizer = optim.Adam(alignNet.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    ae_optimizer = optim.Adam(ae.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    scheduler_optimizer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scheduler_alignNet_optimizer = torch.optim.lr_scheduler.CosineAnnealingLR(alignNet_optimizer, T_max=100)
    scheduler_ae_optimizer = torch.optim.lr_scheduler.CosineAnnealingLR(ae_optimizer, T_max=100)

    ce_loss = nn.CrossEntropyLoss().to(opt.gpu)
    mse_loss = nn.MSELoss().to(opt.gpu)

    import math
    iters = math.ceil(dataset.ntrain / opt.batchsize)
    beta = 0.01
    gamma = 0.0
    coin = 0
    for it in range(start_step, opt.niter + 1):

        if it % iters == 0:
            beta = min(0.02 * (it / iters), 1)
            gamma = min(0.001 * (it / iters), 1)

        blobs = data_layer.forward()
        feat_data = blobs['data']
        labels_numpy = blobs['labels'].astype(int)
        labels = torch.from_numpy(labels_numpy.astype('int')).to(opt.gpu)

        C = np.array([dataset.train_att[i, :] for i in labels])
        C = torch.from_numpy(C.astype('float32')).to(opt.gpu)
        X = torch.from_numpy(feat_data).to(opt.gpu)
        sample_C = torch.from_numpy(np.array([dataset.train_att[i, :] for i in labels.unique()])).to(opt.gpu)
        sample_C_n = labels.unique().shape[0]
        sample_label = labels.unique().cpu()
        # use feature data and train attribute to generate x_mean, z_mu, z_var, z
        x_mean, z_mu, z_var, z = vae(X, C)
        # Standard vae loss function
        vae_loss, ce, kl = vae_loss_function(x_mean, X, z_mu, z_var, z, beta=beta)

        sample_labels = np.array(sample_label)

        # Relabel scattered labels as continuous
        re_batch_labels = []
        for label in labels_numpy:
            index = np.argwhere(sample_labels == label)
            re_batch_labels.append(index[0][0])
        re_batch_labels = torch.LongTensor(re_batch_labels).to(opt.gpu)

        align_loss = 0.
        rec = 0.
        swap_loss = 0.
        x1, z1, semantic_unspecific_first, semantic_matched_first, class_share_first, class_unique_first = ae(x_mean)
        align = alignNet(semantic_matched_first, sample_C)
        align = align.view(-1, labels.unique().cpu().shape[0])
        align_loss = align_loss + ce_loss(align, re_batch_labels)
        rec = rec + mse_loss(x1, X)

        x3, _, _, _, _, _ = ae.forward_swap(x_mean)
        swap_loss = swap_loss + mse_loss(x3, X)

        if opt.twice:
            x2, z2, semantic_unspecific_second, semantic_matched_second, class_share_second, class_unique_second = ae(X)
            align = alignNet(semantic_matched_second, sample_C)
            align = align.view(-1, labels.unique().cpu().shape[0])
            align_loss = align_loss + ce_loss(align, re_batch_labels)
            rec = rec + mse_loss(x2, X)
            x4, _, _, _, _, _ = ae.forward_swap(x_mean)

            swap_loss = swap_loss + mse_loss(x4, X)

        cluster_ids_x1, cluster_centers1 = kmeans(
            X=F.normalize(X), num_clusters=opt.centernum, distance='euclidean', device=torch.device('cuda:0')
        )

        # kmeans = KMeans(n_clusters=opt.centernum, mode='cosine', verbose=1)
        # cluster_ids_x1 = kmeans.fit_predict(F.normalize(x_mean))

        criterion = SupConLoss()
        between_set_loss = criterion(semantic_matched_first, cluster_ids_x1.to(opt.gpu))


        criterion = SupConLoss()
        intra_set_loss = 0
        for i in range(opt.centernum):
            set = cluster_ids_x1 == i
            sample_set = class_unique_first[set]
            label_in_set = labels[set]
            intra_set_loss = intra_set_loss + criterion(sample_set, label_in_set)
        loss = opt.vae * vae_loss + opt.align * align_loss + opt.rec * rec + opt.swap * swap_loss \
               + opt.bet * between_set_loss \
               + opt.intra * intra_set_loss
        optimizer.zero_grad()
        alignNet_optimizer.zero_grad()
        ae_optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        alignNet_optimizer.step()
        ae_optimizer.step()
        # if it > 1000:
        #     scheduler_optimizer.step()
        #     scheduler_relation_optimizer.step()
        #     scheduler_ae_optimizer.step()
        #     scheduler_dis_optimizer.step()
        # withInjihe_optimizer.step()
        if it % opt.disp_interval == 0 and it:
            # log_text = 'Iter-[{}/{}]; loss: {:.3f}; kl:{:.3f}; p_loss:{:.3f}; rec:{:.3f}; cs_loss:{:.3f};tc:{:.3f}; gamma:{:.3f};'.format(it,
            #                                  opt.niter, loss.item(),kl.item(),p_loss.item(),rec.item(),cs_loss.item() ,tc_loss.item(), gamma)

            log_text = 'Iter-[{}/{}]; loss: {:.3f}; vae_loss:{:.3f};align_loss:{:.3f}; rec:{:.3f};swap_loss:{:.3f};between_loss:{:.3f};intra_loss:{:.3f}; '.format(
                it,
                opt.niter, loss.item(), vae_loss.item(), align_loss.item(), rec.item(), swap_loss.item(),
                between_set_loss.item(), intra_set_loss.item())
            log_print(log_text, log_dir)

        if it % opt.evl_interval == 0 and it > 150:  # 15000
            vae.eval()
            ae.eval()
            alignNet.eval()
            gen_feat, gen_label = synthesize_feature_test(vae, ae, dataset, opt)
            with torch.no_grad():
                train_feature = ae.encoder(dataset.train_feature.to(opt.gpu))[:, opt.UNS_dim:].cpu()
                test_unseen_feature = ae.encoder(dataset.test_unseen_feature.to(opt.gpu))[:, opt.UNS_dim:].cpu()
                test_seen_feature = ae.encoder(dataset.test_seen_feature.to(opt.gpu))[:, opt.UNS_dim:].cpu()

            train_X = torch.cat((train_feature, gen_feat), 0)
            train_Y = torch.cat((dataset.train_label, gen_label + dataset.ntrain_class), 0)
            """ZSL"""
            cls = classifier.CLASSIFIER(opt, gen_feat, gen_label, dataset, test_seen_feature, test_unseen_feature,
                                        dataset.ntrain_class + dataset.ntest_class, True, 0.004, 0.5, 50,
                                        opt.nSample, False)
            result_zsl_soft.update(it, cls.acc)
            log_print("ZSL Softmax:", log_dir)
            log_print("Acc {:.2f}%  Best_acc [{:.2f}% | Iter-{}]".format(
                cls.acc, result_zsl_soft.best_acc, result_zsl_soft.best_iter), log_dir)
            """ GZSL"""
            cls = classifier.CLASSIFIER(opt, train_X, train_Y, dataset, test_seen_feature, test_unseen_feature,
                                        dataset.ntrain_class + dataset.ntest_class, True, opt.classifier_lr, 0.5, 20,
                                        opt.nSample, True)

            result_gzsl_soft.update_gzsl(it, cls.acc_unseen, cls.acc_seen, cls.H)

            log_print("GZSL Softmax:", log_dir)
            log_print("U->T {:.2f}%  S->T {:.2f}%  H {:.2f}%  Best_H [{:.2f}% {:.2f}% {:.2f}% | Iter-{}]".format(
                cls.acc_unseen, cls.acc_seen, cls.H, result_gzsl_soft.best_acc_U_T, result_gzsl_soft.best_acc_S_T,
                result_gzsl_soft.best_acc, result_gzsl_soft.best_iter), log_dir)

            if result_gzsl_soft.save_model:
                files2remove = glob.glob(out_dir + '/Best_model_GZSL_*')
                for _i in files2remove:
                    os.remove(_i)
                save_model(it, vae, opt.manualSeed, log_text,
                           out_dir + '/Best_model_GZSL_H_{:.2f}_S_{:.2f}_U_{:.2f}.tar'.format(result_gzsl_soft.best_acc,
                                                                                              result_gzsl_soft.best_acc_S_T,
                                                                                              result_gzsl_soft.best_acc_U_T))

            vae.train()
            ae.train()
            alignNet.train()

        if it % opt.save_interval == 0 and it:
            save_model(it, vae, opt.manualSeed, log_text,
                       out_dir + '/Iter_{:d}.tar'.format(it))
            print('Save model to ' + out_dir + '/Iter_{:d}.tar'.format(it))


if __name__ == "__main__":
    train()
