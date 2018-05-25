import requests
import timer


# http://p8tapdqdq.bkt.clouddn.com/ea22d6a0-58ed-11e8-9594-4cedfb68088f.jpg

#47.95.145.80
#172.16.36.221
#10.1.5.163
def face_aging_web(img_url, choice_img_name):
    url = 'http://172.16.36.218:8081/face_morph/face_aging_web'
    params = {
        'img_url': img_url,
        'choice_img_name': choice_img_name
    }
    return requests.post(url, params=params)

if __name__ == "__main__":
    for i in range(1):
        ti = timer.Timer()
        ti.tic()
        img_url = 'http://p8ypfsng5.bkt.clouddn.com/74e6c512-5b41-11e8-b352-4cedfb68088f.jpg'
        choice_img_name = '86'
        res = face_aging_web(img_url, choice_img_name)
        ti.toc()
        print ti.total_time
        print res.json()
