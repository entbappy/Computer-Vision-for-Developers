import os 
from yoloapp.utils.main_utils import encodeImagetoBase64, decodeImage
from flask import Flask, request, jsonify, render_template, Response 
from flask_cors import CORS, cross_origin 

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predictRoute():
    try:
        image = request.json["image"]
        decodeImage(image, clApp.filename)

        os.system("cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.5 --source ../data/inputImage.jpg")
        opencodedbase64 = encodeImagetoBase64("yolov5/runs/detect/exp/inputImage.jpg")
        result = {"image": opencodedbase64.decode('utf-8')}
        os.system("rm -rf yolov5/runs")
        return jsonify(result)


    except Exception as e:
        return jsonify({"error": str(e)})
    


@app.route("/live", methods=['GET'])
@cross_origin()
def detectLive():
    try:
        os.system("cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.5 --source 0")
        os.system("rm -rf yolov5/runs")
        return "Camera starting!!" 

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host="0.0.0.0", port=8080, debug=True)