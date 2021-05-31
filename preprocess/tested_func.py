import os
import pandas as pd
import time
from functools import wraps

from config.logger import Logging

logger = Logging().log


def func_timer(function):
    '''
    timer
    :param function: counting time consumption
    :return: None
    '''

    @wraps(function)
    def function_timer(*args, **kwargs):
        print('[Function: {name} start...]'.format(name=function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print('[Function: {name} finished, spent time: {time:.6f}s]'.format(name=function.__name__, time=t1 - t0))
        return result

    return function_timer


def read_js(path: str) -> dict:
    """
    批量读取json文件并更新到一个新字典作为输出返回（自动含去重）
    :param path:
    :return:
    """
    subfix = ".json"
    folder = os.listdir(path)
    for obj in folder:
        file = path + obj
        if file.endswith(subfix):
            with open(file, "r", encoding="utf-8") as f:
                yield {obj: eval(f.read())}


def read_file(path: str, batch=1, col="") -> set:
    """
    批量读取txt文件，并返回一个生成器（生成器内的文件自动含去重）
    # 考虑到批量处理时的大文件情况，所以使用生成器形式返回
    :param path: txt文件所再目录
    :param batch: 每次返回的生成器内的文件个数
    :return:
    """
    ret = set()
    count = 0
    folder = os.listdir(path)
    for obj in folder:
        file = path + "/" + obj
        logger.info("processing {file}".format(file=file))
        if file[-3:] == "txt":
            lines = open(file, "r", encoding="utf-8").readlines()
        elif file[-3:] == "csv":
            df = pd.read_csv(file, error_bad_lines=False, index_col=False, lineterminator='\n', dtype='unicode')
            lines = df[col]
        try:
            for line in lines:
                ret.add(line.strip())
        except UnicodeDecodeError:
            logger.warning("UnicodeDecodeError skipped {file}".format(file=file))
            continue
        count += 1
        if count >= batch or obj == folder[-1]:
            for line in ret:
                yield line
            logger.info("length of {file} is {length}".format(file=file, length=len(ret)))
            ret = set()
            count = 0
    return ret


def read_excels(folder: str) -> dict:
    """
    读取excel文件（以xls或xlsx结尾的文件）
    :param folder: excel文件所在的目录
    :return: 依次将excel里面的sheet封装成pandas的dataframe对象，返回{excel_name|sheet_name: sheet_object}
    """
    ret = dict()
    files = os.listdir(folder)
    for suffix in files:
        if suffix.endswith(".xls") or suffix.endswith(".xlsx"):
            for excel_name in files:
                excel = folder + excel_name
                df = pd.ExcelFile(excel)
                sheet_names = df.sheet_names
                for sheet_name in sheet_names:
                    sheet = df.parse(sheet_name)
                    ret.setdefault(excel_name + "|" + sheet_name, sheet)
    return ret


def csv2txt(fout, fin):
    """
    为单个csv文件里面的每个字段生成一个txt文件，csv的字段名为txt的文件名
    :param fout: .txt文件写出路径 /home/data
    :param fin: .csv文件读取路径 /home/data/persona.csv
    :return:
    """
    csvdf = pd.read_csv(fin)
    headers = list(csvdf.columns.values)
    for header in headers:
        path = fout + "/" + header + ".txt"
        with open(path, "w", encoding="utf-8") as fwrite:
            for line in csvdf[header].dropna():
                fwrite.write(line)
                fwrite.write("\n")


def location_loader(loc_path):
    with open(loc_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.replace("\001", "").strip()
            yield line


def repl(table, string):
    # 适用于需要被替换字符多样化，且数量非常多，但复杂度会随着文本的长度增加，150w = 0.2秒
    return "".join(map(lambda x: table[x] if x in table else x, string))

# def get_cell(row):
#     ret = list()
#     n = 0
#     while n < len(row):
#         if row[n].strip():
#             ret.append(row[n])
#         else:
#             try:
#                 s = row[n+1].strip()
#             except IndexError:
#                 return "".join(ret)
#             if s:
#                 ret.append(row[n])
#             else:break
#         n += 1
#     return "".join(ret)


# if __name__=="__main__":
#     folder = "G:/45/"
#     folder2 = "G:/Sample/"
#     c = 0
#     with open(folder2+"45A.txt", "w") as f:
#         for name, value in read_excels(folder).items():
#             excel_name, sheet_name = name.split("|")
#             terms = value["45A"].dropna()
#             for k, v in terms.iteritems():
#                 f.write("".join(v.split("\n")))
#                 f.write("\n\n\n")
#                 c+=1
#         print(c)
