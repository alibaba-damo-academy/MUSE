class ModelFacotry:

    @staticmethod
    def get_model(model_name):
        if model_name in ['dino_r50', 'kang_bench_r50']:
            from .dino_r50 import DINOR50DenseFeatureExtractor
            return DINOR50DenseFeatureExtractor(model_name)
        elif model_name in ['dino_vit-s-16', 'dino_vit-b-16', 'kang_bench_vit-s-16']:
            from .dino_vit import DINOViTDenseFeatureExtractor
            return DINOViTDenseFeatureExtractor(model_name)
        elif model_name == 'uni':
            from .uni import UNIFeatureExtractor
            return UNIFeatureExtractor()
        elif model_name == 'ctrans':
            from .ctranspath import CTransPathFeatureExtractor
            return CTransPathFeatureExtractor()
        elif model_name == 'sup_r50':
            from .sup_r50 import SupR50DenseFeatureExtractor
            return SupR50DenseFeatureExtractor()
        elif model_name in ['sup_vit-s-16', 'sup_vit-b-16']:
            from .sup_vit import SupViTFeatureExtractor
            return SupViTFeatureExtractor(model_name)
        elif model_name in ['mae_vit-b']:
            from .mae import MAEFeatureExtractor
            return MAEFeatureExtractor(model_name)
        elif model_name in ['ibot_vit-s-16', 'ibot_vit-b-16']:
            from .ibot_vit import iBOTViTFeatureExtractor
            return iBOTViTFeatureExtractor(model_name)
        elif model_name in ['dinov2_vit-s', 'dinov2_vit-b', 'dinov2_vit-s-path', 'dinov2_vit-b-path']:
            from .dinov2 import DINOV2ViTDenseFeatureExtractor
            return DINOV2ViTDenseFeatureExtractor(model_name)
        elif model_name == 'conch':
            from .conch_vit import CONCHFeatureExtractor
            return CONCHFeatureExtractor()
        elif model_name == 'gigapath':
            from .prov_gigapath import GigaPathFeatureExtractor
            return GigaPathFeatureExtractor()
        elif model_name == 'chief':
            from .chief import CHIEFPathFeatureExtractor
            return CHIEFPathFeatureExtractor()
        elif model_name in [
            'muse_vit-s-16', 'lfov_muse_vit-s-16',
            'muse_vit-b-16', 'lfov_muse_vit-b-16'
            ]:
            from .muse_vit import MUSEViTFeatureExtractor
            return MUSEViTFeatureExtractor(model_name)
        elif model_name in [
            'muse_r50', 'lfov_muse_r50'
            ]:
            from .muse_r50 import MUSEConvFeatureExtractor
            return MUSEConvFeatureExtractor(model_name)
        else:
            raise NotImplementedError(f'{model_name} is not implemented.')
