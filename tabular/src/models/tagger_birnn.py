"""Vanilla recurrent network model for sequences tagging."""
import torch
import torch.nn as nn
from src.models.tagger_base import TaggerBase
from src.layers.layer_word_embeddings import LayerWordEmbeddings
from src.layers.layer_bilstm import LayerBiLSTM
from src.layers.layer_pooler import LayerPooler
from src.classes.utils import *
from anchor.anchor.utils import perturb_sentence
import numpy as np
import time


class TaggerBiRNN(TaggerBase):
    """TaggerBiRNN is a Vanilla recurrent network model for sequences tagging."""
    def __init__(self, word_seq_indexer, tag_seq_indexer, class_num, batch_size=1, rnn_hidden_dim=50,
                 freeze_word_embeddings=False, dropout_ratio=0.5, rnn_type='LSTM', gpu=-1,
                 latent_dim = 16, pooling_type = 'attention'):
        super(TaggerBiRNN, self).__init__(word_seq_indexer, tag_seq_indexer, gpu, batch_size)
        self.tag_seq_indexer = tag_seq_indexer
        self.class_num = class_num
        self.rnn_hidden_dim = rnn_hidden_dim
        self.freeze_embeddings = freeze_word_embeddings
        self.dropout_ratio = dropout_ratio
        self.rnn_type = rnn_type
        self.gpu = gpu
        self.word_embeddings_layer = LayerWordEmbeddings(word_seq_indexer, gpu, freeze_word_embeddings)
        self.dropout = torch.nn.Dropout(p=dropout_ratio)
        self.latent_dim = latent_dim
        self.birnn_layer = LayerBiLSTM(input_dim=self.word_embeddings_layer.output_dim,
                                       hidden_dim=rnn_hidden_dim,
                                       gpu=gpu)
        self.pooler = LayerPooler(input_dim = self.birnn_layer.output_dim, gpu=gpu, pooling_type = pooling_type)

        if latent_dim is not None:            
            self.dim_red = nn.Sequential(
                nn.Linear(in_features=self.pooler.output_dim, out_features=latent_dim),
                nn.Sigmoid()
            )            
            self.dim_red.apply(self.inititialize_random_projection)
            lin_layer_in = latent_dim
        else:
            lin_layer_in = self.pooler.output_dim            

        self.lin_layer = nn.Linear(in_features=lin_layer_in, out_features=class_num)
        
        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        if gpu >= 0:
            self.cuda(device=self.gpu)
        self.nll_loss = nn.NLLLoss()


    def forward(self, word_sequences):
        mask = self.get_mask_from_word_sequences(word_sequences)
        z_word_embed = self.word_embeddings_layer(word_sequences)
        z_word_embed_d = self.dropout(z_word_embed)
        rnn_output_h = self.birnn_layer(z_word_embed_d, mask)
        pooled_output_h = self.pooler(rnn_output_h, mask)
        if self.latent_dim is not None:
            latent_h = self.dim_red(pooled_output_h)            
        else:
            latent_h = pooled_output_h
        z_rnn_out = self.lin_layer(latent_h) # shape: batch_size x class_num
        y = self.log_softmax_layer(z_rnn_out)        
        return y

    def get_logits(self, word_sequences):
        mask = self.get_mask_from_word_sequences(word_sequences)
        z_word_embed = self.word_embeddings_layer(word_sequences)
        z_word_embed_d = self.dropout(z_word_embed)
        rnn_output_h = self.birnn_layer(z_word_embed_d, mask)
        pooled_output_h = self.pooler(rnn_output_h, mask)
        if self.latent_dim is not None:
            latent_h = self.dim_red(pooled_output_h)            
        else:
            latent_h = pooled_output_h
        z_rnn_out = self.lin_layer(latent_h) # shape: batch_size x class_num
        return z_rnn_out

    def get_latents(self, word_sequences):
        mask = self.get_mask_from_word_sequences(word_sequences)
        z_word_embed = self.word_embeddings_layer(word_sequences)
        z_word_embed_d = self.dropout(z_word_embed)
        rnn_output_h = self.birnn_layer(z_word_embed_d, mask)
        pooled_output_h = self.pooler(rnn_output_h, mask)
        if self.latent_dim is not None:
            latent_h = self.dim_red(pooled_output_h)            
        else:
            latent_h = pooled_output_h
        return latent_h # shape: batch_size x latent_dim


    def initialize_from_pretrained(self, pretrained_path):
        print("Initializing model weights from model at %s" % pretrained_path)
        pretrained_model = torch.load(pretrained_path)
        state_dict = pretrained_model.state_dict()
        
        keys = [key for key in state_dict.keys()]
        for key in keys:
            if key.startswith('lin_layer'):
                del state_dict[key]

        self.load_state_dict(state_dict, strict = False)  


    def inititialize_random_projection(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0, std = 1 / (m.out_features ** 1/2) )

    def get_loss(self, word_sequences_train_batch, tag_sequences_train_batch):        
        outputs_tensor_train_batch_one_hot = self.forward(word_sequences_train_batch)
        targets_tensor_train_batch = self.tag_seq_indexer.items2tensor(tag_sequences_train_batch)
        loss = self.nll_loss(outputs_tensor_train_batch_one_hot, targets_tensor_train_batch)
        return loss


    def saliency_map(self, word_sequence, saliency_type = 'counterfactual', class_id = None, neighbors_obj = None,
                        counterfactual_method = 'unk', language_model = None, tokenizer = None):
        '''
        directional derivative
            what is the gradient of the selected_logit in the direction of the word_embedding
            i.e., how much does the logit change if there were "more" or "less" of this word
            dot_product(embedding_grad, embedding / norm(embedding))
            this is really equivalent to simonyan without the abs value

        counterfactual importance
            for a word, find the avg. difference in the class logit between 
                a) the original input
                b) the input with that word subbed with neighbors in embedding space

        counterfactual_method and language_model used for saliency_type = 'counterfactual'

        each method sets a certain importance_metric, which gets passed to utils.saliency_highlight

        '''

        # very hacky, but since we already have trained models, just set the <unk> vectors to a vector of 0s every time. 
        unk_id = self.word_seq_indexer.item2idx_dict['<unk>']
        self.word_seq_indexer.embedding_vectors_list[unk_id] = torch.zeros(300)
        self.tensor_ensure_gpu(self.word_seq_indexer.embedding_vectors_list[unk_id])

        self.zero_grad()
        self.eval() # .train() run as needed
        start = time.time()

        # go ahead and get the predicted tag
        predicted_tag = self.predict_tags_from_words([word_sequence], constrain_to_classes = None, quiet = True)[0]

        word_sequences = [word_sequence]    
        mask = self.get_mask_from_word_sequences(word_sequences)
        z_word_embed = self.word_embeddings_layer(word_sequences)
        rnn_output_h = self.birnn_layer(z_word_embed, mask)
        mlp_output = self.pooler.mlp(rnn_output_h)

        # attention or serrano saliency
        if saliency_type == 'attention' or saliency_type == 'serrano':                        
            # start with attention
            self.train()
            query = self.pooler.query_vector
            key = mlp_output
            value = rnn_output_h
            d_k = query.size(-1)
            scores = torch.matmul(query, key.transpose(-2, -1)) \
                     / np.sqrt(d_k) 
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = torch.nn.functional.softmax(scores, dim = -1)
            context_embedding = torch.matmul(p_attn, value).view(-1,self.birnn_layer.output_dim)        
            z_rnn_out = self.lin_layer(context_embedding)
            probs = torch.nn.functional.softmax(z_rnn_out, dim=1)
            selected_prob = torch.max(probs) if class_id is None else probs[0,class_id]
            
            # retain and compute gradients
            p_attn.retain_grad()
            selected_prob.backward()

            # get attn_grad and attn*attn_grad
            attn_gradient = p_attn.grad
            attn_gradient = attn_gradient.view(-1)
            p_attn = p_attn.view(-1)
            attn_times_attn_grad = attn_gradient * p_attn

            if saliency_type == 'attention':
                importance_metric = p_attn.detach().cpu().numpy()
            elif saliency_type == 'serrano':
                importance_metric = attn_times_attn_grad.detach().cpu().numpy()

        # li or simonya saliency
        # word importance is abs value of sum of max_logit gradient w.r.t. word embeddings
        # for simonya, multiply gradients by embeddings too
        if saliency_type in ['li','simonyan','directional']:
            self.train()

            # get word embeddings            
            word_sequences = [word_sequence]    
            mask = self.get_mask_from_word_sequences(word_sequences)
            z_word_embed = self.word_embeddings_layer(word_sequences).detach().cpu()

            num_perturbations = 100
            word_embedding_gradients_sum = torch.zeros_like(z_word_embed)
            
            # get smoothed embedding gradients
            for i in range(num_perturbations):

                embedding_noise = torch.zeros_like(z_word_embed)

                # get noise for input
                emb_dim = z_word_embed.size(-1)
                zero_mean = torch.zeros(emb_dim)
                isotropic_noise = .003 * torch.eye(emb_dim)
                noise_object = torch.distributions.MultivariateNormal(zero_mean, isotropic_noise)
                for j in range(len(word_sequence)):
                    embedding_noise[0,j,:] = noise_object.sample() if i > 5 else torch.zeros(emb_dim) # keep a few copies of the original. almost never does the noise have norm near zero.


                # add noise to input and move to gpu
                noisy_word_embed = z_word_embed + embedding_noise
                noisy_word_embed = self.tensor_ensure_gpu(noisy_word_embed)
                noisy_word_embed.requires_grad = True

                # forward pass
                rnn_output_h = self.birnn_layer(noisy_word_embed, mask)
                mlp_output = self.pooler.mlp(rnn_output_h)        
                context_embedding, p_attn = self.pooler.attention(query = self.pooler.query_vector, 
                                key = mlp_output, 
                                value = rnn_output_h, 
                                mask = mask) 
                z_rnn_out = self.lin_layer(context_embedding) 
                selected_logit = torch.max(z_rnn_out) if class_id is None else z_rnn_out[0,class_id]  

                # retain and compute gradients
                # z_word_embed.retain_grad()
                noisy_word_embed.retain_grad()
                selected_logit.backward()

                # get embedding gradients and accumulate to the sum
                word_embedding_gradients = noisy_word_embed.grad.detach().cpu()
                word_embedding_gradients_sum += word_embedding_gradients

            # smoothed gradients
            smoothed_word_embedding_gradients = word_embedding_gradients_sum / num_perturbations

            # compute importance metric for each word
            embedding_gradient_ASV = np.zeros(len(word_sequence)) # li
            dX_times_X_ASV = np.zeros(len(word_sequence)) # simonyan
            directional_grads = np.zeros(len(word_sequence))
            for i in range(len(word_sequence)):
                embedding = z_word_embed[0,i,:]
                embedding_gradient = smoothed_word_embedding_gradients[0,i,:] # 0 is for batching idx
                embedding_times_gradient = embedding * embedding_gradient                            
                embedding_normalized = embedding / embedding.norm(2) # l2 norm
                
                embedding_gradient_ASV[i] = torch.abs(torch.sum(embedding_gradient)).item()
                dX_times_X_ASV[i] = torch.abs(torch.sum(embedding_times_gradient)).item()
                directional_grads[i] = torch.sum(embedding_gradient * embedding_normalized).item()
                    
            # set importance metric according to method
            if saliency_type == 'li':
                importance_metric = embedding_gradient_ASV
            elif saliency_type == 'simonyan':
                importance_metric = dX_times_X_ASV
            elif saliency_type == 'directional':
                importance_metric = directional_grads

    
        # counterfactual importance
        if saliency_type == "counterfactual":

            # get valid_idx for words to sample in expected_word method
            if counterfactual_method == 'expected_word':
                all_vocab = [force_ascii(tokenizer._convert_id_to_token(i)) for i in range(tokenizer.vocab_size)]
                valid_vocab = [word for word in all_vocab if word in self.word_seq_indexer.item2idx_dict]
                valid_idx = np.argwhere([word in self.word_seq_indexer.item2idx_dict for word in all_vocab]).reshape(-1)

            # forward pass
            logits = self.get_logits([word_sequence])
            selected_logit = torch.max(logits) if class_id is None else logits[0,class_id]  
            selected_logit = selected_logit.detach().cpu()

            # need class ids to possibly negate the importance values later
            neg_class_id = self.tag_seq_indexer.item2idx_dict['neg']
            pred_class_id = torch.argmax(logits.view(-1)).item()
            explain_class_id = pred_class_id if class_id is None else class_id

            # get avg. difference in selected_logit and the class logit obtained from perturbed inputs (perturbed at a specific word)
            logit_differences = np.zeros(len(word_sequence))
            for slot_id in range(len(word_sequence)):

                # if 'unk' or 'neighbors', fill the slot with either the <unk> vector or neighboring words in embedding space (according to counterfactual_method)
                if counterfactual_method != 'expected_word':
                    counterfactual_sequences = replace_word(word_sequence, slot_id, neighbors_obj, 
                                                            tagger_word_dict = self.word_seq_indexer.item2idx_dict, method = counterfactual_method)
                    counterfactual_logits = self.get_logits(counterfactual_sequences).detach().cpu()
                    mean_logit = torch.mean(counterfactual_logits[:,explain_class_id])
                    
                    logit_differences[slot_id] = selected_logit - mean_logit

                # if 'expected_word', find the expected logit over p(x_i | x_{-i})
                elif counterfactual_method == 'expected_word':
                    # import ipdb; ipdb.set_trace()
                    expected_logit = expected_score(word_sequence = word_sequence, 
                                                mask_position = slot_id, 
                                                class_id = explain_class_id, 
                                                tagger = self, 
                                                language_model = language_model, 
                                                tokenizer = tokenizer, 
                                                vocab = valid_vocab, 
                                                valid_idx = valid_idx)
                    logit_differences[slot_id] = selected_logit - expected_logit                

            # set importance metric
            importance_metric = logit_differences

        # quick fix so that saliency maps are consistently directional between classes. ^ is always pos, and down-caret is always neg        
        neg_class_id = self.tag_seq_indexer.item2idx_dict['neg']
        explain_class_id = self.tag_seq_indexer.item2idx_dict[predicted_tag] if class_id is None else class_id
        if explain_class_id == neg_class_id and (saliency_type == 'directional' or saliency_type == 'counterfactual'):
            importance_metric = -importance_metric   

        importance_str = saliency_list_values(word_sequence, importance_metric, saliency_type)

        return importance_str


    def explain_instance(self, word_sequence, neighbors_obj, saliency_type = 'counterfactual', counterfactual_method = 'unk', language_model = None, tokenizer = None,
                            decision_boundary = False):
        '''
        explains instance via salience maps on the original input, and salience maps for a nearby input that received the opposite-class prediction

        there's no doubt that many forward passes are done unnecessarily in this method and those it calls. should clean this ups
        '''    
        self.eval()

        # validate input shape
        if type(word_sequence) is str:
            word_sequence = word_sequence.split()

        # get classes and str text        
        classes = self.tag_seq_indexer.idx2items([0,1])
        pos_class_id = 0 if classes[0] == "pos" else 1
        neg_class_id = 1 - pos_class_id
        text = ' '.join(word_sequence)

        # get logits for original input
        orig_logits = self.get_logits([word_sequence]).detach().cpu().squeeze()
        orig_logits_str = "Original evidence margin: %s \n\n" % float_to_signed_str(orig_logits[pos_class_id] - orig_logits[neg_class_id])

        # get tags
        pred = np.argmax(orig_logits)
        predicted_tag = classes[pred]
        opposite_tag = classes[1-pred]

        # get saliency for original input
        orig_explanation = self.saliency_map(word_sequence, neighbors_obj = neighbors_obj, saliency_type = saliency_type,
                                        counterfactual_method = counterfactual_method, language_model = language_model, tokenizer = tokenizer)


        explanation = orig_logits_str + \
                    'Informative values in the input:\n' + \
                    orig_explanation 

        return explanation


    def decision_boundary_explanation(self, word_sequence, neighbors_obj, sufficient_conditions_print = True):
        self.eval()

        # validate input shape, split strs
        if type(word_sequence) is str:
            word_sequence = word_sequence.split()

        # get the path to the nearest input that receives the opposite prediction as word_sequence. (steps being one-word changes)
        path_sequences, path_sequences_highlighted = get_path_to_decision_boundary(perturb_sentence = perturb_sentence, 
                                                                            word_sequence = word_sequence, 
                                                                            tagger = self, 
                                                                            neighbors_obj = neighbors_obj)

        # set up class-margin correspondence for the explanation
        classes = self.tag_seq_indexer.idx2items([0,1])
        pos_class_id = 0 if classes[0] == "pos" else 1
        neg_class_id = 1 - pos_class_id
        # disclaimer = "Positive evidence margins favor class %s, while negative margins favor class %s \n\n" % (classes[0],classes[1])

         # get logits for original input
        orig_logits = self.get_logits([word_sequence]).detach().cpu().squeeze()
        evidence_margin = orig_logits[pos_class_id] - orig_logits[neg_class_id]
        signed_evidence_margin = float_to_signed_str(evidence_margin)
        orig_logits_str = "Original evidence margin: %s" % signed_evidence_margin

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

        # get text str
        text = ' '.join(word_sequence)
        
        # gather the formatted difference in logits and the perturbations
        step_logits_str_list = []
        step_str_list = []
        for idx in steps_to_show:
            step_logits_str = "Evidence margin: %s" % float_to_signed_str(path_logits[idx, pos_class_id] - path_logits[idx, neg_class_id])
            step_logits_str_list.append(step_logits_str)
            step_str_list.append(' '.join(path_sequences_highlighted[idx]))


        # create explanation
        if not sufficient_conditions_print:
            explanation = \
                explanation = "%s \nEdits for changing prediction to \'%s\':" % (orig_logits_str, opposite_tag)

            # add steps to explanation
            for i, idx in enumerate(steps_to_show):
                step_explanation = \
                    "\n\n" + \
                    "Step %d. " % (i+1) + step_logits_str_list[i] +\
                    "\n----\n" + \
                    step_str_list[i]
                explanation += step_explanation

        if sufficient_conditions_print:
            explanation = \
                "Edits for changing prediction to \'%s\':\n" % opposite_tag + \
                step_str_list[-1]

        return explanation