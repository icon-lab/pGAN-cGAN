def train():
    import time
    from options.train_options import TrainOptions
    from data import CreateDataLoader
    from models import create_model
    from util.visualizer import Visualizer
    opt = TrainOptions().parse()
    model = create_model(opt)
    #Loading data
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('Training images = %d' % dataset_size)    
    visualizer = Visualizer(opt)
    total_steps = 0
    #Starts training
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()
            #Save current images (real_A, real_B, fake_B)
            if  epoch_iter % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch,epoch_iter, save_result)
            #Save current errors   
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)
            #Save model based on the number of iterations
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')
    
            iter_data_time = time.time()
        #Save model based on the number of epochs
        print(opt.dataset_mode)
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)
    
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
def test():
    import sys
    sys.argv=args  
    import os
    from options.test_options import TestOptions
    from data import CreateDataLoader
    from models import create_model
    from util.visualizer import Visualizer
    from util import html
    
    
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle

    
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        img_path[0]=img_path[0]+str(i)
        print('%04d: process image... %s' % (i, img_path))
        visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)    
    webpage.save()    
import sys
sys.argv.extend(['--model','pGAN'])
args=sys.argv
if '--training' in str(args):
    train()
else:
    sys.argv.extend(['--serial_batches'])
    test()    