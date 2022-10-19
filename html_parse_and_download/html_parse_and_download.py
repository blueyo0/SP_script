# -*- encoding: utf-8 -*-
'''
@File    :   html_parse_and_download.py
@Time    :   2022/10/14 14:44:32
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   解析html并进行下载，有参考https://zhuanlan.zhihu.com/p/63982089
'''
    
import re
import os
from contextlib import closing
import threading
import requests

def find_all_links_in_html(html_file):
    html_file = open(html_file, "r")
    content = html_file.readlines()
    pattern = "https://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]"
    links = []
    for line in content:
        results = re.findall(pattern, line)
        links += [(r, r.split("/")[-1], r.split("/")[-2]) for r in results]
    return links

def get_link_generator(links):
    for link in links:
        try:
            yield link
        except:
            break
        
headers = {
'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36'
}
timeout = 30
OUT_DIR = "/mnt/petrelfs/wanghaoyu/why/why_download/NHANES_II_Xray"
def download(img_url, img_name, img_class, out_dir=OUT_DIR):
    if os.path.isfile(os.path.join(out_dir, str(img_class), img_name)):
        return    ####如果之前下载过这个文件，就跳过
    with closing(requests.get(img_url, stream=True, headers=headers, timeout=timeout)) as r:
        rc = r.status_code
        if 299 < rc or rc < 200:
            print ('returnCode%s\t%s' % (rc, img_url))
            return
        content_length = int(r.headers.get('content-length', '0'))
        if content_length == 0:
            print ('size0\t%s' % img_url)
            return
        try:
            with open(os.path.join(os.path.join(out_dir, str(img_class)), img_name), 'wb') as f:
                for data in r.iter_content(1024):
                    f.write(data)
        except Exception as e:
            print(e)
            print('savefail\t%s' % img_url)


lock = threading.Lock()
def loop(imgs):
    print ('thread %s is running...' % threading.current_thread().name)

    while True:
        try:
            with lock:
                img_url,img_id,img_class = next(imgs)
                print(img_class)
        except StopIteration:
            break
        try:
            download(img_url, img_id, img_class)
        except:
            print ('exceptfail\t%s' % img_url)
    print ('thread %s is end...' % threading.current_thread().name)
    

def download_links(link_gen, thread_num=100):
    for i in range(0, thread_num):
        t = threading.Thread(target=loop, name='LoopThread %s' % i, args=(link_gen,))
        t.start()



if __name__=="__main__":
    print("hello")
    links = find_all_links_in_html("Index of public_NHANES_X-rays_fullres_.html")
    link_gen = get_link_generator(links)
    
    download_links(link_gen)
    
    links = find_all_links_in_html("Index of public_NHANES_X-rays_marks_.html")
    link_gen = get_link_generator(links)
    
    download_links(link_gen)
