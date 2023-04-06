from Autodataset import AutoDataset


input1 = '/home/gst/14tdisk/14-4tdisk/1a数据总库20221022/识别数据汇总/Lechwe_behavior/Behavior'
# ouput1 = '/home/gst/lj/project/YOLO-learning/auto_mkdataset/TEMP_DATA/output3'
ouput1 = None
config = 'IdObj_config.yaml'
ad = AutoDataset(input1, ouput1, config)
ad.makedataset(datasettype='individual', convert=False, enhance=False)