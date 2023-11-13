"""
Dataset Collection Tool.
Author:		Seunghoon Woo (seunghoonwoo@korea.ac.kr)
Modified: 	December 16, 2020.
"""
import json
import os
import sys
import subprocess
import re
import time
import traceback
from datetime import datetime

import numpy as np
import tlsh  # Please intall python-tlsh
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

"""GLOBALS"""

currentPath = os.getcwd()
gitCloneURLS = currentPath + "/sample"  # Please change to the correct file (the "sample" file contains only 10 git-clone urls)
clonePath = currentPath + "/repo_src/"  # Default path
tagDatePath = currentPath + "/repo_date/"  # Default path
resultPath = currentPath + "/repo_functions/"  # Default path
# ctagsPath = "/usr/local/bin/ctags"  # Ctags binary path (please specify your own ctags path)
ctagsPath = r"D:\ctags\ctags-2023-10-23_p6.0.20231022.0-5-g35abbb1-x86\ctags.exe"

# Generate directories
shouldMake = [clonePath, tagDatePath, resultPath]
for eachRepo in shouldMake:
    if not os.path.isdir(eachRepo):
        os.mkdir(eachRepo)


# Generate TLSH
def computeTlsh(string):
    string = str.encode(string)
    hs = tlsh.forcehash(string)
    return hs


def removeComment(string):
    # Code for removing C/C++ style comments. (Imported from VUDDY and ReDeBug.)
    # ref: https://github.com/squizz617/vuddy
    c_regex = re.compile(
        r'(?P<comment>//.*?$|[{}]+)|(?P<multilinecomment>/\*.*?\*/)|(?P<noncomment>\'(\\.|[^\\\'])*\'|"(\\.|[^\\"])*"|.[^/\'"]*)',
        re.DOTALL | re.MULTILINE)
    return ''.join([c.group('noncomment') for c in c_regex.finditer(string) if c.group('noncomment')])


def normalize(string):
    # Code for normalizing the input string.
    # LF and TAB literals, curly braces, and spaces are removed,
    # and all characters are lowercased.
    # ref: https://github.com/squizz617/vuddy
    return ''.join(string.replace('\n', '').replace('\r', '').replace('\t', '').replace('{', '').replace('}', '').split(
        ' ')).lower()


def hashing(repoPath):
    # This function is for hashing C/C++ functions
    # Only consider ".c", ".cc", and ".cpp" files
    possible = (".c", ".cc", ".cpp")

    fileCnt = 0
    funcCnt = 0
    lineCnt = 0

    resDict = {}

    for path, dir, files in os.walk(repoPath):
        for file in files:
            filePath = os.path.join(path, file)

            if file.endswith(possible):
                try:
                    # Execute Ctgas command
                    functionList = subprocess.check_output(
                        ctagsPath + ' -f - --kinds-C=* --fields=neKSt "' + filePath + '"', stderr=subprocess.STDOUT,
                        shell=True).decode()

                    f = open(filePath, 'r', encoding="UTF-8")

                    # For parsing functions
                    lines = f.readlines()
                    allFuncs = str(functionList).split('\n')
                    func = re.compile(r'(function)')
                    number = re.compile(r'(\d+)')
                    funcSearch = re.compile(r'{([\S\s]*)}')
                    tmpString = ""
                    funcBody = ""

                    fileCnt += 1

                    for i in allFuncs:
                        elemList = re.sub(r'[\t\s ]{2,}', '', i)
                        elemList = elemList.split('\t')
                        funcBody = ""

                        if i != '' and len(elemList) >= 8 and func.fullmatch(elemList[3]):
                            funcStartLine = int(number.search(elemList[4]).group(0))
                            funcEndLine = int(number.search(elemList[7]).group(0))

                            tmpString = ""
                            tmpString = tmpString.join(lines[funcStartLine - 1: funcEndLine])

                            if funcSearch.search(tmpString):
                                funcBody = funcBody + funcSearch.search(tmpString).group(1)
                            else:
                                funcBody = " "

                            funcBody = removeComment(funcBody)
                            funcBody = normalize(funcBody)
                            funcHash = computeTlsh(funcBody)

                            if len(funcHash) == 72 and funcHash.startswith("T1"):
                                funcHash = funcHash[2:]
                            elif funcHash == "TNULL" or funcHash == "" or funcHash == "NULL":
                                continue

                            storedPath = filePath.replace(repoPath, "")
                            if funcHash not in resDict:
                                resDict[funcHash] = []
                            resDict[funcHash].append(storedPath)

                            lineCnt += len(lines)
                            funcCnt += 1

                except subprocess.CalledProcessError as e:
                    print("Parser Error:", e)
                    continue
                except Exception as e:
                    print("Subprocess failed", e)
                    continue

    return resDict, fileCnt, funcCnt, lineCnt


def indexing(resDict, title, filePath):
    # For indexing each OSS

    fres = open(filePath, 'w')
    fres.write(title + '\n')

    for hashval in resDict:
        if hashval == '' or hashval == ' ':
            continue

        fres.write(hashval)

        for funcPath in resDict[hashval]:
            fres.write('\t' + funcPath)
        fres.write('\n')

    fres.close()


def list_dir(path):  # 传入存储的list
    """list child dir @ jyf
    """
    chile_folder_list = []
    for file in os.listdir(path):  # os.listdir(path)，路径下的文件及文件夹，不包含子文件和子文件夹
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):  # 判断是否目录
            chile_folder_list.append(file_path)
    return chile_folder_list


def time_str_to_stamp(time_str: str) -> int:
    """ string to timestamp
    example: time_str = "2022-01-01 12:00:00"
    """
    datetime_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    return int(datetime_obj.timestamp())


def time_stamp_to_str(time_stamp: int) -> str:
    """ timestamp to string
    example: time_stamp: 1613214141
    """
    local_time = time.localtime(time_stamp)
    return str(time.strftime("%Y-%m-%d %H:%M:%S", local_time))


def tags_cluster(tag_time_dict: dict):
    """时间聚类 @ jyf
    """
    n_clusters = 5
    # less than clusters count
    if len(tag_time_dict.items()) < n_clusters:
        return [(k, v) for k, v in tag_time_dict.items()]

    # else
    lst = [(k, time_str_to_stamp(v)) for k, v in tag_time_dict.items()]
    lst = sorted(lst, key=lambda x: x[1])
    timestamps = np.array([item[1] for item in lst]).reshape(-1, 1)
    scaler = StandardScaler()
    timestamps_scaled = scaler.fit_transform(timestamps)

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(timestamps_scaled)
    labels = kmeans.labels_
    clustered_data = [(item[0], item[1], label) for item, label in zip(lst, labels)]
    ver_res_dict = {}
    for item in clustered_data:
        # print(f"Name: {item[0]}, Timestamp: {item[1]}, Cluster Label: {item[2]}")
        if item[2] not in ver_res_dict.keys() or item[1] > ver_res_dict[item[2]][1]:
            ver_res_dict[item[2]] = item
    res = [(v[0], time_stamp_to_str(v[1])) for v in list(ver_res_dict.values())]
    # print(res)
    return res


def run_command(command: str):
    """包装执行命令
    """
    try_time = 3
    has_result = False
    command_result = None
    # 尝试三次
    while not has_result and try_time > 0:
        try:
            command_result = subprocess.check_output(
                command, stderr=subprocess.STDOUT, shell=True)
            # print(command_result)
            has_result = True
        except Exception as e:
            try_time -= 1
            print(" + Try again, try time rest {},  {}".format(try_time, e))
            print(traceback.format_exc())
            time.sleep(5)

    return command_result


def main():
    with open(gitCloneURLS, 'r', encoding="UTF-8") as fp:
        funcDateDict = {}
        # lines = [l.strip('\n\r') for l in fp.readlines()]

        start_time = time.time()
        oss_data_path = r"E:\TPLs"
        oss_folder_path_list = list_dir(oss_data_path)
        for (index, oss_folder_path) in enumerate(oss_folder_path_list):
            # check if already handled ok.
            repoName = os.path.basename(oss_folder_path)

            # for eachUrl in lines:
            #     os.chdir(currentPath)
            # repoName = eachUrl.split("github.com/")[1].replace(".git", "").replace("/",
            #                                                                        "@@")  # Replace '/' -> '@@' for convenience
            print("[+] Processing", repoName)

            # git check_out
            time_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
            tag_pattern = r'tag: ([^,)]+)'

            idx_tag_dict = {}  # 记录id-tag
            tag_time_dict = {}  # 记录时间

            try:
                # cloneCommand = eachUrl + ' ' + clonePath + repoName
                # cloneResult = subprocess.check_output(cloneCommand, stderr=subprocess.STDOUT, shell=True).decode()

                # os.chdir(clonePath + repoName)
                os.chdir(oss_folder_path)
                print(f"cd {oss_folder_path}")
                # tags
                tag_command = "git tag"
                tag_result = run_command(tag_command).decode()

                # branch
                # 分支不一定名称为'master'，需要先通过git branch 获取名称再git checkout -f <主分支名称>
                branch_command = "git branch"
                branch_result = run_command(branch_command).decode()
                tag_master = branch_result.split()[-1].strip()  # * master -> ['*','master'] -> 'master'

                # master
                master_command = 'git log --simplify-by-decoration --pretty="format:%ai %d"'
                master_result = run_command(master_command).decode()
                time_result = master_result.split('\n')

                if len(time_result) > 0:  # master 分支
                    time_match = re.search(time_pattern, time_result[0])
                    time_str = time_match.group(1)
                    tag_time_dict[tag_master] = time_str  # tag_time_dict = {’master‘: '2023-10-22 21:22:57','rc'}

                if len(tag_result) > 0:  # 有其他 tag 的，继续处理，获取更多版本信息
                    # logs
                    data_command = 'git log --tags --simplify-by-decoration --pretty="format:%ai %d"'
                    data_result = run_command(data_command).decode()
                    # get tag commit time
                    for tag_info in data_result.split('\n'):
                        time_match = re.search(time_pattern, tag_info)
                        this_time = time_match.group(1)
                        tags = re.findall(tag_pattern, tag_info)
                        if len(tags) > 0:
                            tag_time_dict[tags[0]] = this_time

                # 如何知道大版本, 同时有一定时间跨度
                # select tag for trainning
                rest_tag_time_list = tags_cluster(tag_time_dict)

                # dateCommand = 'git log --tags --simplify-by-decoration --pretty="format:%ai %d"'  # For storing tag dates
                # dateResult = subprocess.check_output(dateCommand, stderr=subprocess.STDOUT, shell=True).decode()
                # tagDateFile = open(tagDatePath + repoName, 'w')
                # tagDateFile.write(str(dateResult))
                # tagDateFile.close()

                # tagCommand = "git tag"
                # tagResult = subprocess.check_output(tagCommand, stderr=subprocess.STDOUT, shell=True).decode()

                resDict = {}
                fileCnt = 0
                funcCnt = 0
                lineCnt = 0

                for (version_idx, tag_tuple) in enumerate(rest_tag_time_list):
                    tag, tag_time = tag_tuple
                    idx_tag_dict[tag] = tag_time
                    print(f"* handling tag: {tag} {tag_time} {version_idx}/{len(rest_tag_time_list)}")
                    if len(tag) == 0:
                        continue

                    # if tagResult == "":
                    #     # No tags, only master repo
                    #
                    #     resDict, fileCnt, funcCnt, lineCnt = hashing(clonePath + repoName)
                    #     if len(resDict) > 0:
                    #         if not os.path.isdir(resultPath + repoName):
                    #             os.mkdir(resultPath + repoName)
                    #         title = '\t'.join([repoName, str(fileCnt), str(funcCnt), str(lineCnt)])
                    #         resultFilePath = resultPath + repoName + '/fuzzy_' + repoName + '.hidx'  # Default file name: "fuzzy_OSSname.hidx"
                    #
                    #         indexing(resDict, title, resultFilePath)

                    # else:
                    #     for tag in str(tagResult).split('\n'):
                    # Generate function hashes for each tag (version)

                    # checkoutCommand = subprocess.check_output("git checkout -f " + tag, stderr=subprocess.STDOUT,
                    #                                           shell=True)
                    # git checkout 换版本
                    checkout_command = "git checkout -f " + tag
                    checkout_result = run_command(checkout_command)
                    resDict, fileCnt, funcCnt, lineCnt = hashing(oss_folder_path)

                    if len(resDict) > 0:
                        if not os.path.isdir(resultPath + repoName):
                            os.mkdir(resultPath + repoName)
                        title = '\t'.join([repoName, str(fileCnt), str(funcCnt), str(lineCnt)])
                        resultFilePath = resultPath + repoName + '/fuzzy_' + tag + '.hidx'

                        indexing(resDict, title, resultFilePath)

                # write time
                json_str = json.dumps(idx_tag_dict)
                with open(file=tagDatePath + repoName, mode='w', encoding='utf-8') as f:
                    f.write(json_str)

            except subprocess.CalledProcessError as e:
                print("Parser Error:", e)
                continue
            except Exception as e:
                print("Subprocess failed", e)
                continue


""" EXECUTE """
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"total cost : {end_time - start_time}")
