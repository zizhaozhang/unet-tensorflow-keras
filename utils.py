'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-07-03 11:25:40
 * @modify date 2017-07-03 11:25:40
 * @desc [description]
'''

# from keras import backend as K
import numpy as np
from PIL import Image
import os, copy, shutil, json
from skimage import measure

class VIS:
    def __init__(self, save_path):

        self.path = save_path
        # TODO
        self.semantic_label = None

        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
        os.mkdir(self.path)

        self.mean_iu = []
        self.cls_iu = []
        self.score_history = {}
        self.suffix = str(np.random.randint(1000))

        self.palette = Image.open('palette_refer.png').palette

    def save_seg(self, label_im, name, im=None, gt=None):

        seg = Image.fromarray(label_im.astype(np.uint8), mode='P') # must convert to int8 first
        seg.palette = copy.copy(self.palette)
        if gt is not None or im is not None:
            gt = Image.fromarray(gt.astype(np.uint8), mode='P') # must convert to int8 first]
            gt.palette = copy.copy(self.palette)
            im = Image.fromarray(im.astype(np.uint8), mode='RGB')
            I = Image.new('RGB', (label_im.shape[1]*3, label_im.shape[0]))
            I.paste(im,(0,0))
            I.paste(gt,(256,0))
            I.paste(seg,(512,0))
            I.save(os.path.join(self.path, name))
        else:   
            seg.save(os.path.join(self.path, name))   

    def add_sample(self, pred, gt):
        score_mean, score_cls = mean_IU(pred, gt)
        self.mean_iu.append(score_mean)
        self.cls_iu.append(score_cls)
        
        return score_mean

    def compute_scores(self, suffix=0):
        meanIU = np.mean(np.array(self.mean_iu))
        meanIU_per_cls = np.mean(np.array(self.cls_iu), axis=0)
        print ('-'*20)
        print ('overall mean IU: {} '.format(meanIU))
        print ('mean IU per class')
        for i, c in enumerate(meanIU_per_cls):
            print ('\t class {}: {}'.format(i,c))
        print ('-'*20)
        
        data = {'mean_IU': '%.2f' % (meanIU), 'mean_IU_cls': ['%.2f'%(a) for a in meanIU_per_cls.tolist()]}
        self.score_history['%.10d' % suffix] = data
        json.dump(self.score_history, open(os.path.join(self.path, 'meanIU{}.json'.format(self.suffix)),'w'), indent=2, sort_keys=True)

        
def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl   = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)
 
    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_, IU


'''Used by Tensorflow'''
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask

def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl

def extract_masks(segm, cl, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise ValueError('Uneuqal image and mask size')