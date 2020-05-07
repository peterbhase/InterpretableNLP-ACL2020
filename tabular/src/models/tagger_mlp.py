"""Vanilla recurrent network model for sequences tagging."""
import torch
import torch.nn as nn
from src.models.tagger_base import TaggerBase
from src.classes.utils import *
import numpy as np
import time


class TaggerMLP(TaggerBase):
    """TaggerMLP is a Vanilla MLP for vector data."""
    def __init__(self, data_encoder, tag_seq_indexer, class_num, batch_size=1, input_dim=115,
                 dropout_ratio=0, gpu=-1, latent_dim = 16, hidden_dim = 50):
        super(TaggerMLP, self).__init__(data_encoder, tag_seq_indexer, gpu, batch_size)
        self.tag_seq_indexer = tag_seq_indexer
        self.class_num = class_num
        self.input_dim = input_dim
        self.dropout_ratio = dropout_ratio
        self.gpu = gpu
        self.dropout = torch.nn.Dropout(p=dropout_ratio)
        self.latent_dim = latent_dim

        hidden_sizes = [hidden_dim, hidden_dim]

        # feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(in_features=hidden_sizes[0], out_features=hidden_sizes[1]),
            nn.ReLU()
        )

        # (optional) dimension reduction
        if latent_dim is not None:            
            self.dim_red = nn.Sequential(
                nn.Linear(in_features=hidden_sizes[-1], out_features=latent_dim),
                nn.Sigmoid()
            )            
            self.dim_red.apply(self.inititialize_random_projection)
            lin_layer_in = latent_dim
        else:
            lin_layer_in = hidden_sizes[-1]

        # final layer
        self.lin_layer = nn.Linear(in_features=lin_layer_in, out_features=class_num)
        
        # set up quantities for loss
        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        if gpu >= 0:
            self.cuda(device=self.gpu)
        self.nll_loss = nn.NLLLoss()

        # init weights
        self._initialize_weights()


    def forward(self, X_dense):
        X_sparse = self.dense2sparse(X_dense)        
        hidden = self.feature_extractor(X_sparse)
        latents = self.dim_red(hidden) if self.latent_dim is not None else hidden
        logits = self.lin_layer(latents)        
        y = self.log_softmax_layer(logits)        
        return y

    def get_logits(self, X_dense):
        X_sparse = self.dense2sparse(X_dense)        
        hidden = self.feature_extractor(X_sparse)
        latents = self.dim_red(hidden) if self.latent_dim is not None else hidden
        logits = self.lin_layer(latents)    
        return logits

    def get_latents(self, X_dense):
        X_sparse = self.dense2sparse(X_dense)        
        hidden = self.feature_extractor(X_sparse)
        latents = self.dim_red(hidden) if self.latent_dim is not None else hidden
        return latents # shape: batch_size x latent_dim


    def initialize_from_pretrained(self, pretrained_path):
        print("Initializing model weights from model at %s" % pretrained_path)
        pretrained_model = torch.load(pretrained_path)
        state_dict = pretrained_model.state_dict()

        keys = [key for key in state_dict.keys()]
        for key in keys:
            if key.startswith('lin_layer'):
                del state_dict[key]

        self.load_state_dict(state_dict, strict = False)  


    def freeze_unfreeze_parameters(self, epoch, args):
        
        # linear layer
        set_to = (epoch >= args.unfreeze_lin_layer)
        self.lin_layer.weight.requires_grad = set_to
        self.lin_layer.bias.requires_grad = set_to

        # feature extractor layers
        set_to = (epoch >= args.unfreeze_feature_extractor)
        for m in self.feature_extractor.modules():
            if hasattr(m,'weight'):
                m.requires_grad = set_to
            if hasattr(m,'bias'):
                m.requires_grad = set_to


    def _initialize_weights(self, from_sklearn = False, dataset = None):
        # xavier initialization
        for m in self.modules():
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.xavier_uniform_(m.bias.view(-1,1))


    def inititialize_random_projection(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0, std = 1 / (m.out_features ** 1/2) )

    def get_loss(self, word_sequences_train_batch, tag_sequences_train_batch):        
        outputs_tensor_train_batch_one_hot = self.forward(word_sequences_train_batch)
        targets_tensor_train_batch = self.tag_seq_indexer.items2tensor(tag_sequences_train_batch)
        loss = self.nll_loss(outputs_tensor_train_batch_one_hot, targets_tensor_train_batch)
        return loss


    def saliency_explain_instance(self, data_row, counterfactual_method = 'mean'):
        '''
        
        '''    
        self.eval()

        # validate data shape
        data_row = force_array_2d(data_row)

        # fit imputation models if they have not yet been fit
        if not hasattr(self, 'feature_id_to_imputation_model'):
            self.fit_imputation_models(method = counterfactual_method)

        # set up class-margin correspondence for the explanation
        classes = self.tag_seq_indexer.idx2items([0,1])
        pos_class_id = 0 if classes[0] == "above $50K" else 1
        neg_class_id = 1 - pos_class_id

         # get logits for original input
        orig_logits = self.get_logits(data_row).detach().cpu().squeeze()
        orig_logits_str = "Evidence margin: %s \n\n" % float_to_signed_str(orig_logits[pos_class_id] - orig_logits[neg_class_id])
        
        classes = self.tag_seq_indexer.idx2items([0,1])

        # get logits for original input
        orig_logits = self.get_logits(data_row).detach().cpu().squeeze()

        # get predicted tag
        pred = np.argmax(orig_logits)
        predicted_tag = classes[pred]

        # get saliency for original input
        orig_explanation = self.saliency_map(data_row, counterfactual_method = counterfactual_method)

        # make explanation str
        explanation = orig_logits_str + \
            "Informative values in input:\n" + \
            orig_explanation

        return explanation



    def decision_boundary_explanation(self, data_row, sufficient_conditions_print = True):
        '''
        takes an explainer object of type anchor_tabular.explainer
        '''
        self.eval()

        # get the path to the nearest input that receives the opposite prediction as word_sequence. (steps being one-word changes)
        path_sequences, path_sequences_highlighted = get_path_to_decision_boundary(X_orig_dense = data_row, tagger = self)

        # set up class-margin correspondence for the explanation
        classes = self.tag_seq_indexer.idx2items([0,1])
        pos_class_id = 0 if classes[0] == "above $50K" else 1
        neg_class_id = 1 - pos_class_id
        # disclaimer = "Positive evidence margins favor class %s, while negative margins favor class %s \n\n" % (classes[0],classes[1])

        # validate input shape
        data_row = force_array_2d(data_row)

         # get logits for original input
        orig_logits = self.get_logits(data_row).detach().cpu().squeeze()
        orig_logits_str = "Original evidence margin: %s" % float_to_signed_str(orig_logits[pos_class_id] - orig_logits[neg_class_id])

        # get tags
        pred = np.argmax(orig_logits)
        predicted_tag = classes[pred]
        opposite_tag = classes[1-pred]

        # get logits for steps in the path
        path_logits = self.get_logits(path_sequences).detach().cpu()
        num_steps = path_logits.shape[0]

        # if <= 4 steps, then show every step
        if num_steps <= 4:
            steps_to_show = np.arange(num_steps)
        # if > 4 steps, show last 4 steps
        else: 
            steps_to_show = np.arange(num_steps-4,num_steps)
        
        # gather the formatted difference in logits and the perturbations
        step_logits_str_list = []
        for idx in steps_to_show:
            step_logits_str = "Evidence margin: %s" % float_to_signed_str(path_logits[idx, pos_class_id] - path_logits[idx, neg_class_id])
            step_logits_str_list.append(step_logits_str)

        # create explanation
        if not sufficient_conditions_print:
            explanation = "%s \nEdits for changing prediction to \'%s\': \n\n" % (orig_logits_str, opposite_tag)

            # add steps to explanation
            for i, idx in enumerate(steps_to_show):
                step_explanation = \
                    "Step %d. \n" % (i+1) + step_logits_str_list[i] + \
                    '\n' + \
                    path_sequences_highlighted[i] + \
                    ('\n\n' if i != len(steps_to_show)-1 else '')
                explanation += step_explanation

        # create shorter explanation for composite method
        if sufficient_conditions_print:
            explanation = \
                "Edits for changing prediction to \'%s\':\n" % (opposite_tag) + \
                path_sequences_highlighted[-1]

        return explanation