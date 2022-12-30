from types import SimpleNamespace
import json


def get_args(name="default",dict_mode=False):
    name_override = None
    
    args_dict = {"name": 'default',
                "wandb": False,
                "skeleton": {"mode": "bbox",
                             "pointwidth": 0,
                             "pointfade": 0,
                             "linewidth": 2,
                             "linefade": 1,
                             "channel_for_negatives": False,
                             "negative_skeleton_rate": 0,
                             "spline_interp": False,
                             "spline_linearized": 5,
                             "ite_skeleton_p": 0.0,
                             "ite_skeleton_neg": -1,
                             "masked": True},
                "unet": {"block": "ffmmm", #m=MBConv,f=FusedMBConv,u=Unet 
                    "act": "silu",
                    "res_mode": "cat", #cat, add
                    "init_mode": "effecientnetv2",
                    "downscale_mode": "avgpool",
                    "upscale_mode": "bilinear",
                    "input_channels": 4,
                    "num_blocks": 5,
                    "num_c": [8,16,32,48,64],
                    "num_repeat": [1,2,2,4,4],
                    "expand_ratio": [1,4,4,6,6],
                    "SE": [0,0,1,1,1]
                },
                "training": {
                    "pretrain_name_ite": None, #e.g. "mix,100000"
                    "weight_mode_loss": None,
                    "max_size": 128,
                    "augment": False,
                    "aug_rampup": 10000,
                    "max_iter": 800000,
                    "batch": 32,
                    "lr": 1e-4,
                    "recon_mode": "L1", #one of ["L1", "L2", "BCE"]
                    "weight_decay": 1e-5,
                    "use_datasets": "coco",
                    "dataset_p": None,
                }
            }
        
    if name is not None and name!="default":
        if name=="unet":
            pass
        elif name=="test":
            #args_mod = default_extra(1)
            args_mod = {"skeleton": {"mode": "extrema",
                                     "pointwidth": 3,
                                     "pointfade": 2,
                                     "masked": False}}
            args_mod["unet"] = {"num_c": [8,8,16,32,32]}
            args_mod["training"] = {"max_size": 64,
                                    "augment": True,
                                    "aug_rampup": None,
                                    "recon_mode": "BCE"}
        elif name=="amix_no_aug":
            args_mod = default_extra(1)
            args_mod["training"]["augment"] = False
        elif name=="amix":
            args_mod = default_extra(1)
        elif name=="amix256":
            args_mod = default_extra(2)
            args_mod["training"]["max_size"] = 256
            args_mod["training"]["batch"] = 16
        elif name=="bbox":
            args_mod = default_extra(1)
            args_mod["skeleton"]["mode"] = "bbox"
            args_mod["skeleton"]["pointwidth"] = 0
            args_mod["skeleton"]["pointfade"] = 0
            args_mod["skeleton"]["linewidth"] = 2
            args_mod["skeleton"]["linefade"] = 1
            args_mod["skeleton"]["masked"] = False
            args_mod["skeleton"]["spline_interp"] = False
            args_mod["skeleton"]["ite_skeleton_p"] = 0
            args_mod["skeleton"]["negative_skeleton_rate"] = 0
        elif name=="extrema":
            args_mod = default_extra(1)
            args_mod["skeleton"]["mode"] = "extrema"
            args_mod["skeleton"]["pointwidth"] = 4
            args_mod["skeleton"]["pointfade"] = 4
            args_mod["skeleton"]["linewidth"] = 0
            args_mod["skeleton"]["linefade"] = 0
            args_mod["skeleton"]["masked"] = False
            args_mod["skeleton"]["spline_interp"] = False
            args_mod["skeleton"]["ite_skeleton_p"] = 0
            args_mod["skeleton"]["negative_skeleton_rate"] = 0
        elif name=="skeletonize":
            args_mod = default_extra(1)
            args_mod["skeleton"]["mode"] = "skeletonize"
            args_mod["skeleton"]["ite_skeleton_p"] = 0
            args_mod["skeleton"]["negative_skeleton_rate"] = 0
            args_mod["skeleton"]["spline_interp"] = False
            args_mod["skeleton"]["masked"] = False
        elif name=="skeletonize_g":
            args_mod = default_extra(1)
            args_mod["skeleton"]["mode"] = "skeletonize_g"
            args_mod["skeleton"]["ite_skeleton_p"] = 0
            args_mod["skeleton"]["negative_skeleton_rate"] = 0
            args_mod["skeleton"]["spline_interp"] = False
            args_mod["skeleton"]["masked"] = False
        else:
            raise ValueError('Invalid model name')
            
        if name_override is not None:
            args_mod["name"] = name_override
        else:
            args_mod["name"] = name
        
        true_  = lambda *_: True
        for key1, value1 in args_mod.items():
            if type(value1)==dict:
                for key2, value2 in value1.items():
                    if type(value2)==dict:
                        for key3, value3 in value2.items():
                            if true_(args_dict[key1][key2][key3]):
                                args_dict[key1][key2][key3] = value3
                    else:
                        if true_(args_dict[key1][key2]):
                            args_dict[key1][key2] = value2
            else:
                if true_(args_dict[key1]):
                    args_dict[key1] = value1
            
    args = json.loads(json.dumps(args_dict), object_hook=lambda item: SimpleNamespace(**item))
    return args_dict if dict_mode else args

def default_extra(mode):
    if mode==0:
        args_mod = {"skeleton": {"mode": "mix0.5",
                                 "pointwidth": 2,
                                 "pointfade": 0,
                                 "linewidth": 2,
                                 "linefade": 0,
                                 "spline_interp": True,
                                 "negative_skeleton_rate": 0.1,
                                 "ite_skeleton_p": 0.5,
                                 "ite_skeleton_neg": -1}}
    elif mode==1:
        args_mod = {"skeleton": {"mode": "amix0,0,1,1",
                                 "pointwidth": 2,
                                 "pointfade": 0,
                                 "linewidth": 2,
                                 "linefade": 0,
                                 "spline_interp": True,
                                 "negative_skeleton_rate": 0.1,
                                 "ite_skeleton_p": 0.3,
                                 "ite_skeleton_neg": -1},
                    "training": {"use_datasets": "pascal,coco,CHAOS,decathlon",
                                "dataset_p": [1,6,1,2],
                                "augment": True,
                                "aug_rampup": 10000,
                                "recon_mode": "BCE"}}
    elif mode==2:
        args_mod = default_extra(1)
        args_mod["unet"] = {"block": "ffmmmm", #m=MBConv,f=FusedMBConv,u=Unet 
                            "num_blocks": 6,
                            "num_c": [16,32,64,64,128,128],
                            "num_repeat": [1,2,2,4,4,4],
                            "expand_ratio": [1,4,4,6,6,6],
                            "SE": [0,0,1,1,1,1]
                            }
    else:
        raise ValueError("unrecognized default mode")
        
    return args_mod

def update_args(args_namespace):
    args_namespace2 = args_namespace
    true_  = lambda *_: True
    args_mod = get_args(dict_mode=True)
    for key1, value1 in args_mod.items():
        if type(value1)==dict:
            for key2, value2 in value1.items():
                if type(value2)==dict:
                    for key3, value3 in value2.items():
                        try:
                            true_(getattr(getattr(getattr(args_namespace2,key1),key2),key3))
                        except AttributeError:
                            setattr(getattr(getattr(args_namespace2,key1),key2),key3,value3)
                            print("WARNING: added args."+key1+"."+key2+"."+key3+"="+str(value3))
                else:       
                    try:
                        true_(getattr(getattr(args_namespace2,key1),key2))
                    except AttributeError:
                        setattr(getattr(args_namespace2,key1),key2,value2)
                        print("WARNING: added args."+key1+"."+key2+"="+str(value2))
        else:           
            try:
                true_(getattr(args_namespace2,key1))
            except AttributeError:
                setattr(args_namespace2,key1,value1)
                print("WARNING: added args."+key1+"="+str(value1))
    
    return args_namespace2
