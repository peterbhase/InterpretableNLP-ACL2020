"""creates various tagger models"""
import os.path
import torch
from src.models.tagger_birnn import TaggerBiRNN
from src.models.tagger_proto_birnn import TaggerProtoBiRNN
from src.models.tagger_proto_birnn_fixed_dim import TaggerProtoBiRNNFixed


class TaggerFactory():
    """TaggerFactory contains wrappers to create various tagger models."""
    @staticmethod
    def load(checkpoint_fn, gpu=-1):
        if not os.path.isfile(checkpoint_fn):
            raise ValueError('Can''t find tagger in file "%s". Please, run the main script with non-empty \
                             "--save-best-path" param to create it.' % checkpoint_fn)
        tagger = torch.load(checkpoint_fn)
        tagger.gpu = gpu

        tagger.word_seq_indexer.gpu = gpu # hotfix
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
    def create(args, word_seq_indexer, tag_seq_indexer, tag_sequences_train):
        if args.model == 'BiRNN':
            tagger = TaggerBiRNN(word_seq_indexer=word_seq_indexer,
                                 tag_seq_indexer=tag_seq_indexer,
                                 class_num=tag_seq_indexer.get_class_num(),
                                 batch_size=args.batch_size,
                                 rnn_hidden_dim=args.rnn_hidden_dim,
                                 freeze_word_embeddings=args.freeze_word_embeddings,
                                 dropout_ratio=args.dropout_ratio,
                                 rnn_type=args.rnn_type,
                                 gpu=args.gpu,
                                 latent_dim=args.latent_dim,
                                 pooling_type=args.pooling_type)
        elif args.model == 'ProtoBiRNN':
            tagger = TaggerProtoBiRNN(word_seq_indexer=word_seq_indexer,
                                 tag_seq_indexer=tag_seq_indexer,
                                 class_num=tag_seq_indexer.get_class_num(),
                                 batch_size=args.batch_size,
                                 rnn_hidden_dim=args.rnn_hidden_dim,
                                 freeze_word_embeddings=args.freeze_word_embeddings,
                                 dropout_ratio=args.dropout_ratio,
                                 rnn_type=args.rnn_type,
                                 num_prototypes_per_class = args.num_prototypes_per_class,
                                 proto_dim = args.proto_dim,
                                 gpu=args.gpu,
                                 pretrained_path=os.path.join('saved_models','%s.hdf5' % args.pretrained_model),
                                 max_pool_protos=args.max_pool_protos,
                                 similarity_epsilon=args.similarity_epsilon,
                                 hadamard_importance=args.hadamard_importance,
                                 similarity_function_name = args.similarity_function_name
                                 )
        elif args.model == 'ProtoBiRNNFixed':
            tagger = TaggerProtoBiRNNFixed(word_seq_indexer=word_seq_indexer,
                                 tag_seq_indexer=tag_seq_indexer,
                                 class_num=tag_seq_indexer.get_class_num(),
                                 batch_size=args.batch_size,
                                 rnn_hidden_dim=args.rnn_hidden_dim,
                                 freeze_word_embeddings=args.freeze_word_embeddings,
                                 dropout_ratio=args.dropout_ratio,
                                 rnn_type=args.rnn_type,
                                 num_prototypes_per_class = args.num_prototypes_per_class,
                                 proto_dim = args.proto_dim,
                                 gpu=args.gpu,
                                 pretrained_path=os.path.join('saved_models','%s.hdf5' % args.pretrained_model),
                                 max_pool_protos=args.max_pool_protos,
                                 similarity_epsilon=args.similarity_epsilon,
                                 hadamard_importance=args.hadamard_importance,
                                 similarity_function_name = args.similarity_function_name
                                 )
        else:
            raise ValueError('Unknown or misspelled tagger model')
        return tagger
