
from flask import Flask, request, Response
import json
import interface
import config

app = Flask(__name__)

@app.route("/face_morph/face_aging_web",methods=["POST"])
def face_aging_web():
    try:
        img_url = request.args.get('img_url')
        choice_img_name = request.args.get('choice_img_name')
    except Exception as e:
        img_url = ''
        choice_img_name = ''
    if img_url == '' or choice_img_name == '':
        code = 411
    else:
        code, urls = interface.face_switch_interface_url(img_url, choice_img_name)
    print code
    if code == 202:
        result = {
            'status': code,
            'pictureUrls':urls,
        }
    else:
        result = {
            'status': code,
            'pictureUrls':[],
        }
    return Response(json.dumps(result), mimetype='application/json')

if __name__ == "__main__":
    interface.global_init_FaceSwitch()
    app.run(host='172.16.36.218', port=8081, threaded=False, processes=1)
