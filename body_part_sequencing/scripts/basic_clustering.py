import clustering
import argparse 
import numpy as np
import torchvision.datasets as datasets
#import models
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from util import AverageMeter, load_model
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='7donorsForPerdonorclustering', help='path to dataset')
    #parser.add_argument('--data', default='3donorsNoGT', help='path to dataset')
    parser.add_argument('--clustering', type=str, 
                        choices=['Kmeans', 'PIC'], default='Kmeans')
    parser.add_argument('--num_cluster', '--k', type=int, default=35) 
    # 232 is the totoal number of clusters I got for 7 donorsi
    # I'll add 100 for the otjer 3 donors (because the average per donor is 33)
    
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--lr', default=0.05, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum (default: 0.9)')
    parser.add_argument('--batch', default=1, type=int,
                        help='mini-batch size (default: 128)')
    return parser.parse_args()

def compute_features(dataloader, model, N):
    batch_time = AverageMeter()
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux

    return features

def main(args):
    seed = 31
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    ## model
    model = models.alexnet(pretrained=True)
    #model = models.alexnet(pretrained=False)
    model.classifier = list(model.children())[1][:-2]
    #modules=list(model.children())[:-1]
    #model =nn.Sequential(*modules)
    #for p in model.parameters():
    #    p.requires_grad = False
    '''
    model = models.__dict__['alexnet'](sobel=args.sobel)
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    '''
    model.cuda()
    cudnn.benchmark = True

    optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr = args.lr,
            momentum = args.momentum,
            weight_decay = 10**args.wd,
            )

    criterian = nn.CrossEntropyLoss().cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])


    tra = [transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize]

    #model.top_layer = None
    #model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    donors = ['1', '2', '3'] #'0a4', '756', '8bd', '98a', 'ab0', 'b17', 'c4b', 'ccc', 'de9', 'df4']
    cluster_numbers = [20, 20, 20]
    #donors = ['1', '2','3','4','6','7','9'] 
    #cluster_numbers = [112, 78, 97, 65, 52, 43, 61] 
    for index, donor in enumerate(donors):
        dir_name = os.path.join(args.data, donor)
        dataset = datasets.ImageFolder(dir_name , 
                                       transform=transforms.Compose(tra))
        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=args.batch,
            num_workers=1,
            pin_memory=True)


        # get the features for the whole dataset
        features = compute_features(dataloader, model, len(dataset))
        pca_model = PCA(n_components = 64)
        PCAed = pca_model.fit_transform(features)
        kmeans = KMeans(n_clusters = cluster_numbers[index])
        kmeans.fit(PCAed)
        labels = kmeans.predict(PCAed)
        
        with open(donor + "_basicAlexNetClustering.txt", 'w') as fp:
            for ind, label in enumerate(labels):
                new_line = '/home/mousavi/da1/icputrd/arf/mean.js/public/img/' + \
                            dataset.imgs[ind][0].split('/')[-2]+"/"+dataset.imgs[ind][0].split('/')[-1].replace('png','icon.JPG') + \
                            ":" + donor + "_" + str(labels[ind]) + "\n"
                fp.write(new_line) 
        '''
        deepcluster = clustering.__dict__[args.clustering](cluster_numbers[index])
        clustering_loss = deepcluster.cluster(features)
        train_dataset = clustering.cluster_assign(deepcluster.images_lists,
            dataset.imgs)
        with open(donor + "_basicAlexNetImagenet_" + args.clustering + "_clustering.txt", 'w') as fp:
            for assignment in train_dataset.imgs:
                fp.write(assignment[0].replace(dir_name, 
                        '/home/mousavi/da1/icputrd/arf/mean.js/public/img')
                        .replace('png','icon.JPG')
                         + ": " + str(assignment[1]) + "\n")
            
        '''


if __name__ == '__main__':
    args = parse_args()
    main(args)
