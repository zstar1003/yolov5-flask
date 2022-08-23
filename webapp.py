"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
import random

import cv2
import numpy as np
from PIL import Image

import torch
from flask import Flask, render_template, request, redirect

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.plots import plot_one_box

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # print(img_bytes)
        # print("---")
        # print(io.BytesIO(img_bytes))
        # img = cv2.imread(img_bytes)

        # print(img)
        # results = model(img, size=640)

        if img is not None:
            showimg = img
            with torch.no_grad():
                img = letterbox(img, new_shape=imgsz)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.half() if model.half() else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = model(img)[0]

                # Apply NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres)
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], showimg.shape).round()
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(
                                xyxy, showimg, label=label, color=colors[int(cls)], line_thickness=2)

        imgFile = "static/img.jpg"
        cv2.imwrite(imgFile, showimg)
        # image = Image.fromarray(cv2.cvtColor(showimg, cv2.COLOR_BGR2RGB))
        # image.render()  # updates results.imgs with boxes and labels
        # for img in showimg.imgs:
        #     img_base64 = Image.fromarray(img)
        #     img_base64.save("static/image0.jpg", format="JPEG")
        return redirect(imgFile)

    return render_template("index.html")


if __name__ == "__main__":
    weights = 'yolov5s.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = attempt_load(weights, map_location=device)
    model.to(device).eval()
    model.half()
    conf_thres = 0.25  # NMS置信度
    iou_thres = 0.45  # IOU阈值
    # 从模型中获取各类别名称
    names = model.module.names if hasattr(model, 'module') else model.names
    # 给每一个类别初始化颜色
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    stride = int(model.stride.max())
    imgsz = check_img_size(640, s=stride)
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=5000)  # debug=True causes Restarting with stat
