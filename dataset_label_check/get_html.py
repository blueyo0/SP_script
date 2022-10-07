import os, sys, pdb

def get_html(dataset_dir, html_obj):
    sub_dirs = sorted(os.listdir(dataset_dir))
    for sub_dir in sub_dirs:
        sub_root = os.path.join(dataset_dir, sub_dir)
        files = os.listdir(sub_root)
        assert len(files) == 3
        html_obj.write("<td align='center'>{}; Three slides: x, y, z</td><br><hr/>\n".format(sub_root))
        for file_name in files:
            file_path = os.path.join(sub_root, file_name)
            html_obj.write("<img src={} height='{}' width='{}'>\n".format(file_path, 400, 600))
        html_obj.write('<br><br><hr/><hr/>')

if __name__ == '__main__':
    vis_root, html_path = sys.argv[1:]

    html_obj = open(html_path, 'w')
    add_part= '<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />'
    html_obj.write(add_part)
    start_str = "<html><body>\n"
    html_obj.write(start_str)

    dataset_list = sorted(os.listdir(vis_root))
    for dataset_dir in dataset_list:
        get_html(os.path.join(vis_root, dataset_dir), html_obj)

    end_str = "</html></body>\n"
    html_obj.write(end_str)
    html_obj.close()




