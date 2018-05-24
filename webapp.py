
from flask import Flask, request, Response
import json
import interface
import config

app = Flask(__name__)

@app.route("/face_morph/face_aging_web",methods=["POST"])
def face_aging_web():
    try:
        img_url = request.args.get('img_url')
        sex_flag = request.args.get('sex_flag')
    except Exception as e:
        img_url = ''
        sex_flag = ''
    if img_url == '' or sex_flag not in ['female', 'male']:
        code = 411
    else:
        code, urls = interface.face_aging_interface_url(img_url, sex_flag)
    print code
    if code == 202:
        gifPictureUrls = [urls[i] for i in config.gif_indexs]
        result = {
            'status': code,
            'pictureUrls':urls,
            'gifPictureUrls':gifPictureUrls,
        }
    else:
        result = {
            'status': code,
            'pictureUrls':[],
            'gifPictureUrls':[],
        }
    return Response(json.dumps(result), mimetype='application/json')

if __name__ == "__main__":
    interface.global_init()
    app.run(host='172.16.36.221', port=8081, threaded=False, processes=40)
