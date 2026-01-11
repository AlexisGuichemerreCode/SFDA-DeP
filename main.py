import datetime as dt
import sys
from copy import deepcopy

from dlib.process.parseit import parse_input

from dlib.process.instantiators import get_model
from dlib.process.instantiators import get_model_source
from dlib.utils.tools import log_device
from dlib.utils.tools import bye

from dlib.configure import constants
from dlib.learning.train_wsol import Trainer
from dlib.learning.train_sfuda_sdda_wsol import TrainerSdda
from dlib.process.instantiators import get_pretrainde_classifier
from dlib.utils.shared import fmsg
from dlib.utils.shared import is_cc

import dlib.dllogger as DLLogger






def main():
    args, args_dict = parse_input(eval=False)
    log_device(args)

    model, model_src = get_model(args)

    model.cuda(args.c_cudaid)
    if model_src:
        model_src.cuda(args.c_cudaid)

    best_state_dict = deepcopy(model.state_dict())

    inter_classifier = None
    if args.task in [constants.F_CL, constants.NEGEV]:
        if args.sf_uda == False:
            inter_classifier = get_pretrainde_classifier(args)
            inter_classifier.cuda(args.c_cudaid)
        else:
            inter_classifier = model

    if args.task in [constants.STD_CL]:
        if args.sf_uda == True:
            model_src_init = get_model_source(args)
            model_src_init.cuda(args.c_cudaid)
            inter_classifier = model_src_init

            
    if args.sf_uda and args.sdda:
        main_trainer = TrainerSdda

    else:
        main_trainer = Trainer

    trainer = main_trainer(args=args,
                           model=model,
                           classifier=inter_classifier,
                           model_src=model_src
                           )

    DLLogger.log(fmsg("Start epoch 0 ..."))

    trainer.evaluate(epoch=0, split=constants.VALIDSET)
    trainer.model_selection(epoch=0)

    trainer.print_performances()
    trainer.report(epoch=0, split=constants.VALIDSET)

    # if args.ds_to_compute_acc_trainset_source_target in [constants.CAMELYON512, constants.GLAS, constants.CAMELYON17_512]:
    #         if args.measure_loc:
    #             trainer.compute_loc_on_target(0)

    #         trainer.compute_acc_on_target(0)

    DLLogger.log(fmsg("Epoch 0 done."))

    for epoch in range(1, trainer.args.max_epochs + 1, 1):

        DLLogger.log(fmsg(f"Start epoch {epoch} ..."))

        train_performance = trainer.train(
            split=constants.TRAINSET, epoch=epoch)
        trainer.evaluate(epoch, split=constants.VALIDSET)

        if args.entropy_models:
            trainer.update_best_entropy_model(epoch, split=constants.TRAINSET)

        if args.unlearning_models:
            trainer.update_best_unlearning_model(epoch, m_unlearning_models=args.m_unlearning_models)


        #if args.dataset == constants.GLAS and args.cl_train_models:
            #trainer.update_best_cl_train_model(epoch, split=constants.TRAINSET)

        # if args.ds_to_compute_acc_trainset_source_target and epoch % args.cmpt_epoch == 0:
        #     trainer.compute_acc_on_source_and_target(epoch)
        #     trainer.compute_loc_on_source_and_target(epoch)

        trainer.model_selection(epoch=epoch)

        trainer.report_train(train_performance, epoch)
        trainer.print_performances()
        trainer.report(epoch, split=constants.VALIDSET)
        DLLogger.log(fmsg(("Epoch {} done.".format(epoch))))

        trainer.adjust_learning_rate()
        DLLogger.flush()


    splits = [constants.TRAINSET, constants.CLVALIDSET]
    if args.ds_to_compute_acc_trainset_source_target and args.esfda == True:
        PLOT_TASKS = [
            "cl",
            "silhouette",
            "DBI",
            "CH",
            "J_index",
            "f1",
            "precision",
            "recall",
            "image_entropy",
            "acc_normal",
            "acc_cancer",
            "acc_flip",
            "acc_stable",
            "kl_uniform",
            "ECE",
            "NLL",
            "Brier",
        ]

        for split in splits:
            for task in PLOT_TASKS:
                trainer.plot_target_acc_curves(
                    task        = task,
                    cmpt_epoch  = args.cmpt_epoch,
                    split       = split
                )

            # trainer.save_curves(task = "cl", cmpt_epoch = args.cmpt_epoch)
            # trainer.plot_target_acc_curves(task = "cl", cmpt_epoch = args.cmpt_epoch)
            # trainer.plot_target_acc_curves(task = "silhouette", cmpt_epoch = args.cmpt_epoch)
            # trainer.plot_target_acc_curves(task = "DBI", cmpt_epoch = args.cmpt_epoch)
            # trainer.plot_target_acc_curves(task = "CH", cmpt_epoch = args.cmpt_epoch)
            # trainer.plot_target_acc_curves(task = "J_index", cmpt_epoch = args.cmpt_epoch)
            # trainer.plot_target_acc_curves(task = "loc", cmpt_epoch = args.cmpt_epoch)
            # trainer.plot_target_acc_curves(task = "f1", cmpt_epoch = args.cmpt_epoch)
            # trainer.plot_target_acc_curves(task = "precision", cmpt_epoch = args.cmpt_epoch)
            # trainer.plot_target_acc_curves(task = "recall", cmpt_epoch = args.cmpt_epoch)
            # trainer.plot_target_acc_curves(task = "image_entropy", cmpt_epoch = args.cmpt_epoch)
            # trainer.plot_target_acc_curves(task = "acc_normal", cmpt_epoch = args.cmpt_epoch)
            # trainer.plot_target_acc_curves(task = "acc_cancer", cmpt_epoch = args.cmpt_epoch)


        #trainer.plot_source_target_loc_curves(cmpt_epoch = args.cmpt_epoch)  

    # if args.esfda:
    #     trainer.save_metrics(filename="metrics_history.pickle")
    #     trainer.save_loss(filename="loss_history.pickle")
    #     trainer.save_loss_esfda()
    #     trainer.save_unlearning_acc(filename="unlearning_acc_history.pickle")
    #     #trainer.plot_unlearning_acc()
    #     trainer.plot_losses()

        
    if args.sf_uda:
        if args.shot or args.cdcl or args.sfde:
            trainer.save_pseudo_labels()

    if args.entropy_models:
        trainer.save_best_entropy_models()
    
    # if args.unlearning_models:
    #     trainer.save_unlearning_models()

    if args.cl_train_models:
        trainer.save_best_cl_train_models(criterion=constants.CLVALIDSET)
    if args.measure_loc:
        trainer.save_best_cl_train_models(criterion=constants.PXVALIDSET)

    trainer.save_checkpoints()

    trainer.save_best_epoch()
    trainer.capture_perf_meters()

    DLLogger.log(fmsg("Final epoch evaluation on test set ..."))

    if args.task != constants.SEG:
        chpts = [constants.BEST_CL]

        if args.localization_avail:
            chpts = [constants.BEST_LOC] + chpts
    else:
        chpts = [constants.BEST_LOC]

    use_argmax = False

    for eval_checkpoint_type in chpts:
        t0 = dt.datetime.now()

        if eval_checkpoint_type == constants.BEST_LOC:
            epoch = trainer.args.best_loc_epoch
        elif eval_checkpoint_type == constants.BEST_CL:
            epoch = trainer.args.best_cl_epoch
        else:
            raise NotImplementedError

        DLLogger.log(
            fmsg('EVAL TEST SET. CHECKPOINT: {}. ARGMAX: {}'.format(
                eval_checkpoint_type, use_argmax)))

        trainer.load_checkpoint(checkpoint_type=eval_checkpoint_type)

        trainer.evaluate(epoch, split=constants.TESTSET,
                         checkpoint_type=eval_checkpoint_type,
                         fcam_argmax=use_argmax)

        trainer.print_performances(checkpoint_type=eval_checkpoint_type)
        trainer.report(epoch, split=constants.TESTSET,
                       checkpoint_type=eval_checkpoint_type)
        trainer.save_performances(
            epoch=epoch, checkpoint_type=eval_checkpoint_type)

        trainer.switch_perf_meter_to_captured()

        tagargmax = f'Argmax: {use_argmax}'

        DLLogger.log("EVAL time TESTSET - CHECKPOINT {} {}: {}".format(
            eval_checkpoint_type, tagargmax, dt.datetime.now() - t0))
        DLLogger.flush()

    trainer.save_args()
    trainer.plot_perfs_meter()
    bye(trainer.args)


if __name__ == '__main__':
    main()
