import roboflow

rf = roboflow.Roboflow(api_key="*")

project = rf.workspace("*").project("*")
version = project.version(1)
dataset = version.download(model_format="yolov8")