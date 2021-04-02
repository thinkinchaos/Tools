import os
import cv2
import logging
from collections import OrderedDict

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.engine.defaults import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
import detectron2.utils.comm as comm

TRAIN_PATH = './datasets/adas_4_3/train'
VAL_PATH = './datasets/adas_4_3/val'

TRAIN_JSON = './datasets/adas_4_3/train.json'
VAL_JSON = './datasets/adas_4_3/val.json'

TRAIN_FLAG = True
TEST_FROM_VAL = False
TEST_FROM_VIDEO = False

DATASET_CATEGORIES = [
    {"color": [0, 41, 223], "isthing": 1, "id": 1, "name": "roads"},
    {"color": [135, 14, 236], "isthing": 1, "id": 2, "name": "ground_mark"},
    {"color": [224, 0, 249], "isthing": 1, "id": 3, "name": "zebra-crs"},
    {"color": [187, 43, 91], "isthing": 1, "id": 4, "name": "vehicle"},
    {"color": [160, 107, 0], "isthing": 1, "id": 5, "name": "non-motor"},
    {"color": [178, 191, 0], "isthing": 1, "id": 6, "name": "person"},
    {"color": [110, 180, 66], "isthing": 1, "id": 7, "name": "sign"},
]

PREDEFINED_SPLITS_DATASET = {"mlabel_train": (TRAIN_PATH, TRAIN_JSON), "my_val": (VAL_PATH, VAL_JSON), }


def register_dataset_instances(name, metadate, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco",
                                  **metadate)


def get_dataset_instances_meta():
    thing_ids = [k["id"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_dataset():
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key,
                                   metadate=get_dataset_instances_meta(),
                                   json_file=json_file,
                                   image_root=image_root)


# 注册数据集和元数据
def plain_register_dataset():
    DatasetCatalog.register("mlabel_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH, "mlabel_train"))
    MetadataCatalog.get("mlabel_train").set(thing_classes=["pos", "neg"],
                                                    json_file=TRAIN_JSON,
                                                    image_root=TRAIN_PATH)
    DatasetCatalog.register("my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "my_val"))
    MetadataCatalog.get("my_val").set(thing_classes=["pos", "neg"],
                                                json_file=VAL_JSON,
                                                image_root=VAL_PATH)


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-env-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    cfg = get_cfg()  # 拷贝default config副本
    cfg.merge_from_file(args.config_file)  # 从config file-tools 覆盖配置
    cfg.merge_from_list(args.opts)  # 从CLI参数 覆盖配置
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    register_dataset()  # 注册数据集

    if TRAIN_FLAG:
        args = default_argument_parser().parse_args()
        launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank,
               dist_url=args.dist_url,
               args=(args,))

    elif TEST_FROM_VAL:
        GET_VAL_INFO = False
        GET_METRICS = False
        DRAW_CURVES = True
        from detectron2.utils.visualizer import sx_Visualizer_analyse_data

        if GET_VAL_INFO:

            cfg = get_cfg()
            YAML_FILE = "./configs/my_configs/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
            cfg.merge_from_file(YAML_FILE)
            # TRAINED_WEIGHT = "../trained_results/msrcnn_x_101_32x8d_fpn_3x_2020_1_17/model_final.pth"
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
            print('loading from: {}'.format(cfg.MODEL.WEIGHTS))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0  # 分数阈值不能设为0，因为此时超了，是ROI所有的，即100????
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
            cfg.DATASETS.TEST = ("my_val",)
            predictor = DefaultPredictor(cfg)
            dataset_dicts = load_coco_json(VAL_JSON, VAL_PATH, "my_val")

            pred_dicts = []
            gt_dicts = []
            image_idx = 0
            for d in dataset_dicts:
                img = cv2.imread(d["file_name"])
                visualizer = sx_Visualizer_analyse_data(img[:, :, ::-1], metadata=MetadataCatalog.get("my_val"), scale=1)
                gt_instances = visualizer.get_info_from_dataset_dict(d)
                pred_instances = visualizer.get_info_from_predictions(predictor(img)["instances"].to("cpu"))
                for i in range(len(pred_instances)):
                    dict_tmp = {'image_idx':image_idx, 'name':pred_instances[i][0], 'score':pred_instances[i][1], 'bbox':pred_instances[i][2]}
                    pred_dicts.append(dict_tmp)
                for i in range(len(gt_instances)):
                    dict_tmp = {'image_idx':image_idx, 'name':gt_instances[i][0], 'score':gt_instances[i][1], 'bbox':gt_instances[i][2]}
                    gt_dicts.append(dict_tmp)
                image_idx +=1

            import json
            with open("gt.json", "w") as f:
                json.dump(gt_dicts, f)
                print("write 'gt.json' finish")
            with open("pred.json", "w") as f:
                json.dump(pred_dicts, f)
                print("write 'pred.json' finish")

        elif GET_METRICS:

            def compute_IOU(rec1, rec2):
                left_column_max = max(rec1[0], rec2[0])
                right_column_min = min(rec1[2], rec2[2])
                up_row_max = max(rec1[1], rec2[1])
                down_row_min = min(rec1[3], rec2[3])
                if left_column_max >= right_column_min or down_row_min <= up_row_max:  # 两矩形无相交区域的情况
                    return 0
                else:  # 两矩形有相交区域的情况
                    S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
                    S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
                    S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
                    return S_cross / (S1 + S2 - S_cross)

            import json
            with open("gt.json", 'r') as gt_file:
                gt_instances = json.load(gt_file)
            with open("pred.json", 'r') as pred_file:
                pred_instances = json.load(pred_file)

            gt_image_idxes = [dict['image_idx'] for dict in gt_instances if dict['image_idx'] >= 0]
            pred_image_idxes = [dict['image_idx'] for dict in pred_instances if dict['image_idx'] >= 0]
            image_num = 0
            for image_idx in gt_image_idxes:
                if image_idx >= image_num:
                    image_num = image_idx
            # print(image_num)

            classes_list = ['roads', 'ground_mark', 'zebra-crs', 'vehicle', 'non-motor', 'person', 'sign']

            for iou_thresh in range(50, 54 + 1, 5):
                iou_thresh *= 0.01

                metric_pairs = []
                for score_thresh in range(0, 99 + 1, 1):
                    score_thresh *= 0.01
                    APs = []
                    ARs = []
                    FPPIs = []

                    for class_idx in range(len(classes_list)):
                        tp = 0
                        fp = 0
                        gt_num_this_class = 0
                        for img_idx in range(image_num):
                            gts_this_img_class = [dict for dict in gt_instances if dict['name'] == classes_list[class_idx] and dict['image_idx'] == img_idx]
                            preds_this_img_class = [dict for dict in pred_instances if dict['name'] == classes_list[class_idx] and dict['image_idx'] == img_idx]
                            preds_adopted = [dict for dict in preds_this_img_class if dict['score'] > score_thresh]

                            gt_num_this_class += len(gts_this_img_class)

                            for pred_adopted in preds_adopted:
                                max_iou = 0
                                max_iou_gt_idx = 0
                                for gt_idx in range(len(gts_this_img_class)):
                                    iou_tmp = compute_IOU(gts_this_img_class[gt_idx]['bbox'], pred_adopted['bbox'])
                                    if iou_tmp > max_iou:
                                        max_iou = iou_tmp
                                        max_iou_gt_idx = gt_idx

                                if max_iou < iou_thresh:
                                    fp += 1
                                elif max_iou >= iou_thresh:
                                    tp += 1

                        # print(classes_list[class_idx], gt_num_this_class, tp, tp + fp)
                        if (tp + fp) == 0:  # 如果在这个阈值条件下，对这个类别没检出任何目标，舍弃该组数据（不利于曲线分析）
                            gt_num_this_class -= len(gts_this_img_class)
                            # APs.append(0)
                            # ARs.append(0)
                            # FPPIs.append(0)
                            continue

                        AP_this_class = tp / gt_num_this_class
                        AR_this_class = tp / (tp + fp)
                        FPPI_this_class = fp / (tp + fp)

                        if AP_this_class > 1:
                            AP_this_class = 1
                        if AR_this_class > 1:
                            AR_this_class = 1
                        if FPPI_this_class > 1:
                            FPPI_this_class = 1

                        APs.append(AP_this_class)
                        ARs.append(AR_this_class)
                        FPPIs.append(FPPI_this_class)

                    mAP_this_score_thresh = sum(APs) / (len(APs))
                    mAR_this_score_thresh = sum(ARs) / (len(ARs))

                    mFPPI_this_score_thresh = sum(FPPIs) / (len(FPPIs))
                    mMissRate_this_score_thresh = 1 - mAR_this_score_thresh

                    print("IoU_th is {0},  Score_th is {1}, mAP is {2:.3f}, mAR is {3:.3f}, mFPPI is {4:.3f}"
                          .format(iou_thresh, score_thresh,
                                  mAP_this_score_thresh, mAR_this_score_thresh, mFPPI_this_score_thresh))

                    metric_pair = {'iou':iou_thresh, 'score':score_thresh, 'AP':mAP_this_score_thresh,
                                   'AR':mAR_this_score_thresh,'FPPI':mFPPI_this_score_thresh,
                                   'MR':mMissRate_this_score_thresh}
                    metric_pairs.append(metric_pair)

                with open("metric_pairs.json", "w") as f:
                    json.dump(metric_pairs, f)
                    print("write 'metric_pairs.json' finish")

        elif DRAW_CURVES:
            import json
            with open("metric_pairs.json", 'r') as metric_pairs:
                metric_pairs = json.load(metric_pairs)

            score_threshes = []
            precisions = []
            recalls = []
            fppis = []
            mrs = []
            for dict in metric_pairs:
                score_threshes.append(dict['score'])
                precisions.append(dict['AP'])
                recalls.append(dict['AR'])
                fppis.append(dict['FPPI'])
                mrs.append(dict['MR'])

            import math
            # fppis = [math.log(a) for a in fppis]

            import matplotlib.pyplot as plt
            plt.figure(1)
            plt.title('MR-FPPI Curve IoU=0.5')
            plt.xlabel('FPPI')
            plt.ylabel('MR')
            plt.plot(fppis, mrs)
            plt.savefig('./MR-FPPI-IoU05.png')

            plt.figure(2)
            plt.title('score-mAP Curve IoU=0.5')
            plt.xlabel('score_thresh')
            plt.ylabel('mAP')
            plt.plot(score_threshes, precisions)
            plt.savefig('./score-mAP-IoU05.png')

            plt.figure(3)
            plt.title('score-mAR Curve IoU=0.5')
            plt.xlabel('score_thresh')
            plt.ylabel('mAR')
            plt.plot(score_threshes, precisions)
            plt.savefig('./score-mAR-IoU05.png')

            plt.figure(4)
            plt.title('P-R Curve IoU=0.5')
            plt.xlabel('Precison')
            plt.ylabel('Recall')
            plt.plot(precisions, recalls)
            plt.savefig('./P-R-IoU05.png')

        elif TEST_FROM_VIDEO:
            # from detectron2.ML.visualizer import Visualizer
            from detectron2.utils.visualizer import sx_Visualizer

            cfg = get_cfg()
            YAML_FILE = "./configs/my_configs/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
            cfg.merge_from_file(YAML_FILE)
            # TRAINED_WEIGHT = "../trained_results/msrcnn_x_101_32x8d_fpn_3x_2020_1_17/model_final.pth"
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
            print('loading from: {}'.format(cfg.MODEL.WEIGHTS))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set the testing threshold for this model
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
            cfg.DATASETS.TEST = ("my_val",)
            predictor = DefaultPredictor(cfg)

            cap = cv2.VideoCapture("../mydata/test-env.mp4")
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            dstSize = (int(frameWidth/2), int(frameHeight/2))
            total_frame_num_VIDEO = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            out = cv2.VideoWriter("../mydata/DEMO.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, dstSize)

            # print(total_frame_num_VIDEO)
            for frame_idx in range(total_frame_num_VIDEO-1):
                ret, frame = cap.read()
                img = cv2.resize(frame, dstSize)
                visualizer = sx_Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("my_val"), scale=1, instance_mode=ColorMode.SEGMENTATION)
                # visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("my_val"), scale=1, instance_mode=ColorMode.SEGMENTATION)
                # vis = visualizer.draw_dataset_dict(d) # show dataset label
                vis = visualizer.draw_instance_predictions(predictor(img)["instances"].to("cpu"))  # show detection result
                show_img = vis.get_image()[:, :, ::-1]
                cv2.imshow('show', show_img)
                cv2.waitKey(1)
                # print('process frame:', frame_idx)
                out.write(show_img)
            out.release()

