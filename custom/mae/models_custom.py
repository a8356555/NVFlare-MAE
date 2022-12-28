from torchvision import models
import torch
import torch.nn as nn

import mae.models_vit as models_vit
from mae.util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
from utils.parse_arg import DefaultArgs

class StaticticalDataEncoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.encoder_layer1 = nn.Linear(args.other_feature_size, args.hidden_size)
        self.elu1 = nn.ELU()
        self.encoder_layer2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.elu2 = nn.ELU()
        
    def forward(self, x):
        x = self.elu1(self.encoder_layer1(x))
        x = self.elu2(self.encoder_layer2(x))
        return x
    
class HeterogeneousModel(nn.Module):
    def __init__(self, backbone, args=None) -> None:
        super().__init__()
        if args is None:
            args = DefaultArgs()
            
        self.image_backbone = backbone
        if 'resnet' in args.model:
            image_embedding_size = self.image_backbone.fc.in_features
            self.image_backbone.fc = nn.Identity()
        elif 'efficient' in args.model:
            image_embedding_size = self.image_backbone.classifier[1].in_features
            self.image_backbone.classifier = nn.Identity()
        elif 'vit' in args.model:
            image_embedding_size = self.image_backbone.head.in_features
            self.image_backbone.head = nn.Identity()
        self.image_embedding_projector = nn.Linear(image_embedding_size, args.hidden_size)
        self.elu1 = nn.ELU()
        
        self.combine_method = args.combine_method
        self.statistical_data_encoder = StaticticalDataEncoder(args)
        if self.combine_method == 'multiply':
            self.fc = nn.Linear(args.hidden_size, args.nb_classes)
        else:
            self.fc = nn.Linear(args.hidden_size*2, args.nb_classes)
        
    def forward(self, x):
        images, statistical_data = x
        image_embedding = self.image_backbone(images)
        image_embedding = self.elu1(self.image_embedding_projector(image_embedding))
        
        statistical_embedding = self.statistical_data_encoder(statistical_data)
        if self.combine_method == 'multiply':
            fusion = image_embedding*statistical_embedding
        else:
            fusion = torch.concat([image_embedding, statistical_embedding], axis=1)
        return self.fc(fusion)
        

def _get_arg_model(args):
    if 'vit' in args.model:
        model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
            img_size=args.crop_size
        )
    
        if args.finetune and not args.eval:
            checkpoint = torch.load(args.finetune, map_location='cpu')

            print("Load pre-trained checkpoint from: %s" % args.finetune)
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            if args.global_pool:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            else:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

            # manually initialize fc layer
            trunc_normal_(model.head.weight, std=2e-5)
    else:
        # torchvision model
        model = getattr(models, args.model)(pretrained=True)
        if 'resnet' in args.model:
            embedding_size = model.fc.in_features
            model.fc = torch.nn.Linear(embedding_size, args.nb_classes)
        elif 'efficient' in args.model:
            embedding_size = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(embedding_size, args.nb_classes)
        
    return model

def build_model(args):
    model = _get_arg_model(args)
       
    if args.heterogeneous:
        model = HeterogeneousModel(model, args=args)
        
    return model