import logging
import os
import torch
import torchvision

from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from torch import distributed
from utils.torch_transforms import ResizeAndPad


def has_chinese(string):
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


class DDJCallBackVerification(object):
    def __init__(self, val_target, data_dir, vis_dir=None, summary_writer=None, image_size=448, wandb_logger=None,
                 dist=True, vis=False, registry_path=None):
        self.vis = vis
        self.rank: int = distributed.get_rank() if dist else 0
        self.val_dataset_name = val_target
        self.highest_mAP = 0.
        self.highest_acc1 = 0.
        self.val_datasets = []
        if self.rank == 0:
            self.init_dataset(registry_path, recognition_set_path=data_dir, image_size=image_size)

        self.vis_dir = vis_dir if vis_dir else os.path.join(data_dir, '../vis')
        self.summary_writer = summary_writer
        self.wandb_logger = wandb_logger

    def ver_test(self, backbone: torch.nn.Module, global_step: int, debug=False):
        CMCs, mAPs = [], []
        for dataset in self.val_datasets:
            dataset_name = dataset[0].root.split('/')[-2]
            CMC, mAP = test(dataset, backbone, self.vis_dir + '/' + dataset_name, vis=self.vis)
            print(f'[{dataset_name}]mAP: {mAP}')
            print(f'[{dataset_name}]CMC: {CMC[:10]}')
            if self.vis:
                plot_cmc(CMC, dataset_name, self.vis_dir)
            CMCs.append(CMC[:10])
            mAPs.append(mAP)
        CMC = np.mean(CMCs, axis=0)
        mAP = np.mean(mAPs)
        acc1 = CMC[0]
        acc5 = CMC[4]
        print(f'[{self.val_dataset_name}]mAP: {mAP}', flush=True)
        print(f'[{self.val_dataset_name}]CMC: {CMC[:10]}', flush=True)
        if self.summary_writer:
            self.summary_writer.add_scalar(tag=self.val_dataset_name + "mAP", scalar_value=mAP,
                                           global_step=global_step, )
            self.summary_writer.add_scalar(tag=self.val_dataset_name + "Acc1", scalar_value=acc1,
                                           global_step=global_step, )
            self.summary_writer.add_scalar(tag=self.val_dataset_name + "Acc5", scalar_value=acc5,
                                           global_step=global_step, )
        if self.wandb_logger:
            self.wandb_logger.log({
                f'Acc/val-Acc1 {self.val_dataset_name}': acc1,
                f'Acc/val-Acc5 {self.val_dataset_name}': acc5,
                f'Acc/val-mAP {self.val_dataset_name}': mAP,
            })
        if mAP > self.highest_mAP:
            self.highest_mAP = mAP
        if acc1 > self.highest_acc1:
            self.highest_acc1 = acc1
        logging.info(
            '[%s][%d]mAP-Highest: %1.5f' % (self.val_dataset_name, global_step, self.highest_mAP))
        logging.info(
            '[%s][%d]Acc1-Highest: %1.5f' % (self.val_dataset_name, global_step, self.highest_acc1))
        return acc1

    def init_dataset(self, registry_path, recognition_set_path, image_size):
        for dataset in os.listdir(recognition_set_path):
            if os.path.isdir(os.path.join(recognition_set_path, dataset)):
                dataset_name = dataset
                self.val_datasets.append(load_dataset(os.path.join(registry_path, dataset_name), os.path.join(recognition_set_path, dataset), image_size))

    def __call__(self, num_update, backbone: torch.nn.Module, vis=False):
        self.vis = vis
        acc1s = []
        if self.rank == 0 and num_update >= 0:
            backbone.eval()
            acc1 = self.ver_test(backbone, num_update)
            acc1s.append(acc1)
            backbone.train()
        return np.mean(acc1s)


@torch.no_grad()
def load_dataset(registry_set_path, recognition_set_path , image_size=448):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    registry_set = torchvision.datasets.ImageFolder(registry_set_path, transform=transform)
    recognition_set = torchvision.datasets.ImageFolder(recognition_set_path, transform=transform)
    return registry_set, recognition_set


# test on Meican Dataset, return CMC and mAP
@torch.no_grad()
def test(data_set, backbone, vis_dir=None, majority_vote=False, vis=False, registry_embeddings=None):
    registry_set, recognition_set = data_set
    backbone.eval()
    registry_labels, recognition_labels, similarity_matrix, indices = extract_embeddings(registry_set, recognition_set, backbone)
    registry_labels = np.array(registry_labels)
    recognition_labels = np.array(recognition_labels)
    
    class_to_idx = None
    if len(recognition_set.classes) > len(registry_set.classes):
        class_to_idx = recognition_set.class_to_idx
    else:
        class_to_idx = registry_set.class_to_idx
    # 对齐两个数据集的类别
    recognition_labels = np.array([class_to_idx[recognition_set.classes[i]] for i in recognition_labels])
    registry_labels = np.array([class_to_idx[registry_set.classes[i]] for i in registry_labels])

    # compute cosine similarity
    # similarity_matrix, indices = cosine_sim_slice(recognition_embeddings, registry_embeddings)
    # visualize top-5 results
    if vis:
        visualize(recognition_set, registry_set, recognition_labels, registry_labels, similarity_matrix, indices,
                  vis_dir=vis_dir)
    CMC, mAP = compute_metrics(registry_labels, recognition_labels, recognition_set, registry_set, indices, similarity_matrix)
    return CMC, mAP


def visualize(recognition_set, registry_set, recognition_labels, registry_labels, similarity_matrix, indices,
              vis_dir=None, debug=False):
    # Create directories to save the visualization results if they don't exist
    save_dir_right = os.path.join(vis_dir, "right")
    save_dir_wrong = os.path.join(vis_dir, "wrong")
    for dir_path in [save_dir_right, save_dir_wrong]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    for i, recognition_idx in tqdm(enumerate(range(len(recognition_set))), desc="Visualizing top-5 results", ncols=100,
                                   total=len(recognition_set)):
        top1_index = indices[i][0]
        recognition_img_path, recognition_label = recognition_set.imgs[recognition_idx]
        top1_label = registry_set.classes[registry_labels[top1_index]]
        gt_label = recognition_set.classes[recognition_label]
        # remove chinese in top1_label and gt_label
        top1_label = top1_label.split('_')[0]
        gt_label = gt_label.split('_')[0]
        if top1_label == gt_label:
            continue
        if debug and i >= 100: break
        recognition_img = Image.open(recognition_img_path)

        # Create a new image to save the visualization results
        width = recognition_img.width * 8 + 5 * 9
        height = recognition_img.height + 5 * 2
        result_img = Image.new("RGB", (width, height), color="white")

        # Paste the recognition image at the leftmost position
        result_img.paste(recognition_img, (5, 5))

        # Draw the top-1 label on the recognition image
        draw = ImageDraw.Draw(result_img)

        # Get the color to draw the text (considering the image's background color)
        recognition_img = recognition_img.convert("RGB")
        bg_color = recognition_img.getpixel((0, 0))
        text_color = "white" if sum(bg_color) > 382 else "black"

        # Draw top-1 label on the image
        text_width, text_height = draw.textsize(f"Top-1: {top1_label}, GT: {gt_label}", font=None)
        draw.rectangle((0, 0, text_width + 10, text_height + 10), fill="black")
        draw.text((5, 5), f"Top-1: {top1_label}, GT: {gt_label}", fill=text_color)

        # Paste the top-5 registry images with a 10-pixel gap between each
        for j, registry_idx in enumerate(indices[i][:5]):
            registry_img_path, _ = registry_set.imgs[registry_idx]
            registry_img = Image.open(registry_img_path)
            result_img.paste(registry_img, (recognition_img.width * (j + 1) + 5 * (j + 1), 5))

        # Paste the gt registry image at the rightmost position
        gt_registry_idx = np.where(registry_labels == recognition_label)[0][0]
        gt_registry_img_path, _ = registry_set.imgs[gt_registry_idx]
        gt_registry_img = Image.open(gt_registry_img_path)
        result_img.paste(gt_registry_img, (recognition_img.width * 6 + 5 * 6, 5))

        # Save the visualization result image to the appropriate folder
        # get the image name of top1
        top1_img_name = registry_set.imgs[top1_index][0]
        result_img.save(os.path.join(save_dir_wrong, f"recognition_{i + 1}_top1_wrong.jpg"))
        msg = str(
            i + 1) + "\t\twrong img: " + recognition_img_path + "\t\ttop1: " + top1_img_name + "\t\ttop1_label:" + top1_label + "\t\tgt: " + gt_label
        # write msg to .txt file
        with open(os.path.join(vis_dir, "wrong_imgs.txt"), 'a') as f:
            f.write(msg + '\n')


def l2_norm(input, axis=1):
    norm = np.linalg.norm(input, 2, axis, True)
    output = np.divide(input, norm)
    return output


def l2_norm_torch(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def extract_embeddings(registry_set, recognition_set, backbone):
    registry_embeddings, registry_labels = None, None
    # extract registry embeddings
    registry_dataloader = torch.utils.data.DataLoader(registry_set, batch_size=16, shuffle=False, num_workers=8,
                                                    drop_last=False)
    registry_embeddings = None
    registry_labels = []
    registry_name = registry_set.root
    for i, (img, label) in tqdm(enumerate(registry_dataloader),
                                desc=f"Extracting {registry_name} registry embeddings", ncols=120,
                                total=len(registry_dataloader)):
        img = img.cuda()
        embedding = backbone(img)
        if registry_embeddings is None:
            registry_embeddings = embedding
        else:
            registry_embeddings = torch.cat((registry_embeddings, embedding), 0)
        registry_labels.append(label.cpu().numpy())
    registry_embeddings = l2_norm_torch(registry_embeddings)
    registry_labels = np.concatenate(registry_labels, axis=0)

    # extract recognition embeddings
    recognition_dataloader = torch.utils.data.DataLoader(recognition_set, batch_size=16, shuffle=False, num_workers=8,
                                                         drop_last=False)
    recognition_embeddings = None
    recognition_labels = []
    recognition_name = recognition_set.root

    for i, (img, label) in tqdm(enumerate(recognition_dataloader),
                                desc=f"Extracting {recognition_name} recognition embeddings", ncols=120,
                                total=len(recognition_dataloader)):
        img = img.cuda()
        embedding = backbone(img)
        if recognition_embeddings is None:
            recognition_embeddings = embedding
        else:
            recognition_embeddings = torch.cat((recognition_embeddings, embedding), 0)
        recognition_labels.append(label)
        
    recognition_embeddings = l2_norm_torch(recognition_embeddings)
    recognition_labels = np.concatenate(recognition_labels, axis=0)

    torch.cuda.empty_cache()
    similarity_matrix = torch.mm(recognition_embeddings, registry_embeddings.T).cpu().numpy()
    del recognition_embeddings, registry_embeddings
    torch.cuda.empty_cache()
    print(f"Similarity matrix shape: {similarity_matrix.shape}, calculating top-100 indices...")
    indices = np.argsort(-similarity_matrix, axis=1)[:, :100]

    return registry_labels, recognition_labels, similarity_matrix, indices


# compute CMC and mAP
def compute_metrics(registry_labels, recognition_labels, recognition_set, registry_set, indices, similarity_matrix, debug=False):
    CMC = np.zeros(100) if debug else np.zeros(len(registry_labels))
    APs = []
    for i in tqdm(range(len(recognition_set)), desc="Computing CMC and mAP", ncols=100):
        # get the ground truth label
        gt_label = recognition_labels[i]
        # get the top-100 labels
        top100_labels = registry_labels[indices[i]]
        # compute CMC
        for j, label in enumerate(top100_labels):
            if label == gt_label:
                CMC[j:] += 1
                break
        # compute AP
        pos = np.where(top100_labels == gt_label)[0]
        if len(pos) == 0:
            AP = 0
        else:
            AP = 0
            for j in range(len(pos)):
                AP += (j + 1) / (pos[j] + 1)
            AP /= len(pos)
        APs.append(AP)
    mAP = np.mean(APs)
    CMC = CMC / len(recognition_set)

    return CMC, mAP


def plot_cmc(CMC, dataset_name, vis_dir, rank_num=10):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, rank_num + 1), CMC[:rank_num], c='red', lw=1)
    plt.xlim(1, rank_num)
    plt.ylim(0, 1)
    plt.xticks(range(1, rank_num + 1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('Rank')
    plt.ylabel('Matching Rate')
    # add text to each point
    for i in range(rank_num):
        plt.text(i + 1, CMC[i], f'{CMC[i]:.2f}', ha='center', va='bottom', fontsize=10)
    plt.grid(linestyle='--', linewidth=1)
    plt.title(f'CMC Curve of {dataset_name}')
    plt.savefig(os.path.join(vis_dir, f'cmc_{dataset_name}.png'))
    plt.close()


def plot_mAP(mAP, dataset_name, vis_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(mAP) + 1), mAP, c='red', lw=1)
    plt.xlim(1, len(mAP) + 1)
    plt.ylim(0, 1)
    plt.xticks(range(1, len(mAP) + 1))
    # plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('Rank')
    plt.ylabel('mAP')
    plt.grid(linestyle='--', linewidth=1)
    plt.title(f'mAP Curve of {dataset_name}')
    plt.savefig(os.path.join(vis_dir, f'mAP_{dataset_name}.png'))
    plt.close()


def normalize(A):
    lengths = (A**2).sum(axis=1, keepdims=True)**.5
    return A/lengths


def cosine_sim(feature1, feature2):
        return np.inner(feature1, feature2) / np.dot(np.linalg.norm(feature1, axis=1, keepdims=True),
                                                    np.linalg.norm(feature2, axis=1, keepdims=True).T)


def cosine_sim_torch(feature1, feature2):
    # feature1 and feature2 are numpy arrays, convert them into torch tensors
    feature1 = torch.from_numpy(feature1).cuda()
    feature2 = torch.from_numpy(feature2).cuda()
    # normalize the feature vectors
    feature1 = normalize(feature1)
    feature2 = normalize(feature2)
    # compute the cosine similarity
    sim = torch.mm(feature1, feature2.T)
    return sim.cpu().numpy()


def cosine_sim_slice(feature1, feature2, slice_size=1000):
    sim = np.zeros((feature1.shape[0], feature2.shape[0]))
    indices = np.zeros((feature1.shape[0], 100), dtype=int)
    for i in tqdm(range(0, feature1.shape[0], slice_size), desc="Computing cosine similarity", ncols=100):
        for j in range(0, feature2.shape[0], slice_size):
            sim[i:i + slice_size, j:j + slice_size] = cosine_sim_torch(feature1[i:i + slice_size], feature2[j:j + slice_size])
            indices[i:i + slice_size] = np.argsort(-sim[i:i + slice_size], axis=1)[:, :100]
    return sim, indices
