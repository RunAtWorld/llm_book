# yolo11

> 仓库参考： https://github.com/ultralytics/ultralytics
> 文档参考： https://docs.ultralytics.com/zh/models/yolo11/#performance-metrics

## 训练
```
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="coco8.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
```
> 下载 onnx需要依赖
> ```
> pip install onnx==1.12.0 
> pip install onnxslim 
> pip install onnxruntime
> ```

## 推理
```
yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
```

## 标注

安装labelimg
```
pip install pyqt5 lxml
```

从 GitHub 仓库克隆并安装
```
git clone https://github.com/tzutalin/labelImg.git
cd labelImg
python labelImg.py
```

```
conda install pyqt=5
conda install -c anaconda lxml
pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py
python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

## 实验
> 参考 https://blog.csdn.net/weixin_44779079/article/details/142676560