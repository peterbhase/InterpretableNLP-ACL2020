"""creates various tagger models"""
import os.path
import torch
from src.models.tagger_mlp import TaggerMLP
from src.models.tagger_proto_mlp import TaggerProtoMLP


class TaggerFactory():
    """TaggerFactory contains wrappers to create various tagger models."""
    @staticmethod
    def load(checkpoint_fn, gpu=-1):
        if not os.path.isfile(checkpoint_fn):
            raise ValueError('Can''t find tagger in file "%s". Please, run the main script with non-empty \
                             "--save-best-path" param to create it.' % checkpoint_fn)
        tagger = torch.load(checkpoint_fn)
        tagger.gpu = gpu

        # an amalgman of hotfixes to get everything on the same gpu
        tagger.tag_seq_indexer.gpu = gpu # hotfix
        if hasattr(tagger, 'char_embeddings_layer'):# very hot hotfix
            tagger.char_embeddings_layer.char_seq_indexer.gpu = gpu # hotfix
        if hasattr(tagger, 'word_embeddings_layer'):
            tagger.word_embeddings_layer.gpu = gpu
        # reset gpu tags 
        for m in tagger.modules():
            if hasattr(m, 'gpu'):
                m.gpu = gpu
        tagger.self_ensure_gpu()
        return tagger


    @staticmethod
    def create(args, data_encoder, tag_seq_indexer):
        if args.model == 'MLP':
            tagger = TaggerMLP(data_encoder=data_encoder,
                                 tag_seq_indexer=tag_seq_indexer,
                                 class_num=tag_seq_indexer.get_class_num(),
                                 batch_size=args.batch_size,
                                 input_dim = args.input_dim,
                                 dropout_ratio=args.dropout_ratio,
                                 gpu=args.gpu,
                                 latent_dim=args.latent_dim,
                                 hidden_dim=args.hidden_dim)
        elif args.model == 'ProtoMLP':
            tagger = TaggerProtoMLP(data_encoder=data_encoder,
                                 tag_seq_indexer=tag_seq_indexer,
                                 class_num=tag_seq_indexer.get_class_num(),
                                 batch_size=args.batch_size,
                                 input_dim = args.input_dim,
                                 dropout_ratio=args.dropout_ratio,
                                 num_prototypes_per_class = args.num_prototypes_per_class,
                                 proto_dim = args.proto_dim,
                                 gpu=args.gpu,
                                 hidden_dim=args.hidden_dim,
                                 pretrained_path=os.path.join('saved_models','%s.hdf5' % args.pretrained_model),
                                 max_pool_protos=args.max_pool_protos,
                                 similarity_epsilon=args.similarity_epsilon,
                                 hadamard_importance=args.hadamard_importance,
                                 similarity_function_name = args.similarity_function_name
                                 )
        else:
            raise ValueError('Unknown or misspelled tagger model')
        return tagger
