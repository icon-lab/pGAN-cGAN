def create_model(opt):
    model = None
    print(opt.model)
    #Set dataset mode and load models based on the selected model (pGAN or cGAN)
    if opt.model == 'cGAN':
        opt.dataset_mode = 'unaligned_mat'
        from .cgan_model import cGAN
        model = cGAN()
    elif opt.model == 'pGAN':
        opt.dataset_mode = 'aligned_mat'
        from .pgan_model import pGAN
        model = pGAN()        
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    #Initizlize the model based on the arguments 
    model.initialize(opt)
    print("model %s was created" % (model.name()))
    return model
