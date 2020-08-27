import utils
from utils import checkattr

##-------------------------------------------------------------------------------------------------------------------##

## Function for defining classifier model
def define_classifier(args, config, device):
    # -import required model
    from models.classifier import Classifier
    # -create model
    if (hasattr(args, "depth") and args.depth>0):
        model = Classifier(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            # -conv-layers
            conv_type=args.conv_type, depth=args.depth, start_channels=args.channels, reducing_layers=args.rl,
            num_blocks=args.n_blocks, conv_bn=True if args.conv_bn=="yes" else False, conv_nl=args.conv_nl,
            global_pooling=checkattr(args, 'gp'),
            # -fc-layers
            fc_layers=args.fc_lay, fc_units=args.fc_units, h_dim=args.h_dim,
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn=="yes" else False, fc_nl=args.fc_nl, excit_buffer=True,
            # -training related parameters
            AGEM=utils.checkattr(args, 'agem')
        ).to(device)
    else:
        model = Classifier(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            # -fc-layers
            fc_layers=args.fc_lay, fc_units=args.fc_units, h_dim=args.h_dim,
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn=="yes" else False, fc_nl=args.fc_nl, excit_buffer=True,
            # -training related parameters
            AGEM=utils.checkattr(args, 'agem')
        ).to(device)
    # -return model
    return model

##-------------------------------------------------------------------------------------------------------------------##

## Function for (re-)initializing the parameters of [model]
def init_params(model, args):
    # - reinitialize all parameters according to default initialization
    model.apply(utils.weight_reset)
    # - initialize parameters according to chosen custom initialization (if requested)
    if hasattr(args, 'init_weight') and not args.init_weight=="standard":
        utils.weight_init(model, strategy="xavier_normal")
    if hasattr(args, 'init_bias') and not args.init_bias=="standard":
        utils.bias_init(model, strategy="constant", value=0.01)
    # - use pre-trained weights in conv-layers?
    if utils.checkattr(args, "pre_convE") and hasattr(model, 'depth') and model.depth>0:
        load_name = model.convE.name if (
            not hasattr(args, 'convE_ltag') or args.convE_ltag=="none"
        ) else "{}-{}".format(model.convE.name, args.convE_ltag)
        utils.load_checkpoint(model.convE, model_dir=args.m_dir, name=load_name)
    return model

##-------------------------------------------------------------------------------------------------------------------##