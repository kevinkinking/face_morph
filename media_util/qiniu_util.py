import config
from qiniu import Auth, put_file, put_data
import uuid

access_key = 'TPww7ooVRKMZz1RGg0EFDh0SVRwVZ9TImR42DFHw'
secret_key = 'CSnmVfgg2I1k4Z0EZjImvIr3R8nq8gsgIz9t_uZh'

base_url = 'http://p8ypfsng5.bkt.clouddn.com/'


q = Auth(access_key, secret_key)

bucket_name = 'babygrow'


def qiniu_upload_file(file_path, ext=config.img_format):
    uuid_str = str(uuid.uuid1())
    key = uuid_str + ext
    token = q.upload_token(bucket_name, key, 3600)
    ret, info = put_file(token, key, file_path)
    if ret['key'].encode('utf-8') == key:
        return base_url + key
    else:
        return ''

def qiniu_upload_data(data, ext=config.img_format):
    uuid_str = str(uuid.uuid1())
    key = uuid_str + ext
    token = q.upload_token(bucket_name, key, 3600)
    ret, info = put_data(token, key, data)
    if ret is None:
    	return ''
    ret_key = ret['key'].encode('utf-8')
    return base_url + ret_key

# print qiniu_upload_file('/home/bbt/2.jpg')

# if is_py2:
#     assert ret['key'].encode('utf-8') == key
# elif is_py3:
#     assert ret['key'] == key

# assert ret['hash'] == etag(localfile)