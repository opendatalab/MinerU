
def join_path(*args):
    return '/'.join(str(s).rstrip('/') for s in args)


def get_top_percent_list(num_list, percent):
    """
    获取列表中前百分之多少的元素
    :param num_list:
    :param percent:
    :return:
    """
    if len(num_list) == 0:
        top_percent_list = []
    else:
        # 对imgs_len_list排序
        sorted_imgs_len_list = sorted(num_list, reverse=True)
        # 计算 percent 的索引
        top_percent_index = int(len(sorted_imgs_len_list) * percent)
        # 取前80%的元素
        top_percent_list = sorted_imgs_len_list[:top_percent_index]
    return top_percent_list


def mymax(alist: list):
    if len(alist) == 0:
        return 0  # 空是0， 0*0也是0大小q
    else:
        return max(alist)


def parse_bucket_key(s3_full_path: str):
    """
    输入 s3://bucket/path/to/my/file.txt
    输出 bucket, path/to/my/file.txt
    """
    s3_full_path = s3_full_path.strip()
    if s3_full_path.startswith("s3://"):
        s3_full_path = s3_full_path[5:]
    if s3_full_path.startswith("/"):
        s3_full_path = s3_full_path[1:]
    bucket, key = s3_full_path.split("/", 1)
    return bucket, key
