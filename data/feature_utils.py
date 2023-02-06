import pathlib
import pandas as pd
from collections import Counter
import csv
import random
import traceback
import numpy as np
import math
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.metrics.pairwise import cosine_similarity
from simhash import Simhash
import re
import Levenshtein
import copy
import string


# import cuml
# from cudf import Series
# get top thousand
def get_top_k(mat, k):
    mat_flat = mat.reshape(-1).tolist()
    # k = 1000
    if k > len(mat_flat):
        k = len(mat_flat)
    mat_flat.sort(reverse=True)
    return mat_flat[:k]


# https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
def id_generator(size=3, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))



# get delimiter of a csv file
# https://stackoverflow.com/a/69796836
# auto detect delimiter for a csv file
def get_delimiter(file_path, bytes=4096):
    sniffer = csv.Sniffer()
    data = open(file_path, "r").read(bytes)
    delimiter = sniffer.sniff(data).delimiter
    return delimiter


# input table
# output set of table header list
def get_header_set(t):
    header_list = t.columns.values.tolist()
    return set(header_list)


# input header set of two table
# output similarity score of feature#1 -- column/row header
def sim_header(header_set_1, header_set_2):
    return len(header_set_1 & header_set_2) / float(len(header_set_1 | header_set_2))


# input: two dataframe
# output row header similarities
# description: 1. if both numberic header 2.if one of 3.if both not
def sim_row_header(t1, t2):
    if t1.index.is_numeric() and t2.index.is_numeric():
        return 0.5
    elif t1.index.is_numeric() or t2.index.is_numeric():
        return 0
    else:
        return sim_header(set(t1.index.tolist()), set(t2.index.tolist()))


# input data frame
# output: numeric_count,empty_count,str_count
def get_cell_type_count(t):
    numeric_count = 0
    empty_count = 0
    str_count = 0
    for rowIndex, row in t.iterrows():  # iterate over rows
        for columnIndex, value in row.items():
            if type(value) == str:
                str_count += 1
            elif type(value) == int or type(value) == float:
                numeric_count += 1
            elif type(value) == None:
                str_count += 1
            else:
                pass
    #                 print(value)
    return [numeric_count, empty_count, str_count]


# input two list of cell_type return by function`get_cell_type_count`
# output: similarity score
def sim_tpye(type_list_1, type_list_2):
    diff_numeric = abs(type_list_1[0] - type_list_2[0])
    diff_empty = abs(type_list_1[1] - type_list_2[1])
    diff_str = abs(type_list_1[2] - type_list_2[2])
    sum_numeric = abs(type_list_1[0] + type_list_2[0])
    sum_empty = abs(type_list_1[1] + type_list_2[1])
    sum_str = abs(type_list_1[2] + type_list_2[2])
    return 1.00 - float(diff_numeric + diff_empty + diff_str) / float(sum_numeric + sum_empty + sum_str)


# use when file is excel file, csv file do not have style feature included
# input: workbook and current sheet(usually the 0th sheet in the workbook when excel file have only one sheet in ML dataset)
# input warning: default sheet index = 0, but might need adjust by different structure of dataset
# output：counter item(dict obj) of 0. background color 1. font color 2.font type 3.font stype 4.top border 5.bottom border 6.left border 7.right border
def get_styles_feature(book, sheet_index=0):
    sheet = book.sheet_by_index(sheet_index)
    background_color_list = []
    font_color_list = []
    font_type_list = []
    font_style_list = []
    top_border_style_list = []
    bottom_border_style_list = []
    left_border_style_list = []
    right_border_stype_list = []
    rows, cols = sheet.nrows, sheet.ncols
    for row in range(rows):
        for col in range(cols):
            thecell = sheet.cell(row, col)
            xfx = sheet.cell_xf_index(row, col)
            xf = book.xf_list[xfx]
            if (not xf.background):
                background_color_list.append(None)
            else:
                background_color_list.append(book.colour_map.get(xf.background.background_colour_index))
            font = book.font_list[xf.font_index]
            if not font:
                font_color_list.append(None)
            else:
                font_color_list.append(book.colour_map.get(font.colour_index))
            #         print(font)
            font_type_list.append(font.family)
            if font.bold:
                font_style_list.append('bold')
            if font.italic:
                font_style_list.append('italic')
            border_obj = xf.border
            top_border_style_list.append(border_obj.top_line_style)
            bottom_border_style_list.append(border_obj.bottom_line_style)
            left_border_style_list.append(border_obj.left_line_style)
            right_border_stype_list.append(border_obj.right_line_style)
    return [Counter(background_color_list), Counter(font_color_list), Counter(font_type_list), Counter(font_style_list),
            Counter(top_border_style_list), Counter(bottom_border_style_list), Counter(left_border_style_list),
            Counter(right_border_stype_list)]


# input two list return from `get_styles_feature`
# output list length of 8 of sim score of:
# 0. background color 1. font color 2.font type 3.font stype 4.top border 5.bottom border 6.left border 7.right border
def sim_style_features(style_feature_1_list, style_feature_2_list):
    return_sim_list = []
    for fea_index in range(len(style_feature_1_list)):
        fea_1_dict = dict(style_feature_1_list[fea_index])
        fea_2_dict = dict(style_feature_2_list[fea_index])
        if (len(fea_1_dict) == 0 and len(fea_2_dict) == 0):
            sim = 0.5
        elif (len(fea_1_dict) == 0 or len(fea_2_dict) == 0):
            sim = 0
        else:
            union_keys = fea_1_dict.keys() | fea_2_dict.keys()
            sim_diffs = 0
            sim_sums = 0
            for key in union_keys:
                list_1_key_val = fea_1_dict.get(key)
                if list_1_key_val is None:
                    list_1_key_val = 0
                list_2_key_val = fea_2_dict.get(key)
                if list_2_key_val is None:
                    list_2_key_val = 0
                sim_diffs += abs(list_1_key_val - list_2_key_val)
                sim_sums += abs(list_1_key_val + list_2_key_val)
            sim = 1 - sim_diffs / float(sim_sums)
        return_sim_list.append(sim)
    return return_sim_list


# dataset handler
# input: dataset dir of the two file
# return: similarity matrix: length 10
def data_dir_handler(dir_1, dir_2, delimiter_1=None, delimiter_2=None):
    df, df2 = None, None
    has_style_feature = False
    if pathlib.Path(dir_1).suffix == '.csv' and pathlib.Path(dir_2).suffix == '.csv':
        print('both csv file, no style feature')
        if delimiter_1 is not None:
            df = pd.read_csv(dir_1, delimiter=delimiter_1)
        else:
            df = pd.read_csv(dir_1)
        if delimiter_2 is not None:
            df2 = pd.read_csv(dir_2, delimiter=delimiter_2)
        else:
            df2 = pd.read_csv(dir_2)
    elif pathlib.Path(dir_1).suffix == '.csv' and pathlib.Path(dir_2).suffix == '.xls':
        print('file#1 is csv, file#2 is xls')
        if delimiter_1 is not None:
            df = pd.read_csv(dir_1, delimiter=delimiter_1)
        else:
            df = pd.read_csv(dir_1)
        df2 = pd.read_excel(dir_2)
    elif pathlib.Path(dir_1).suffix == '.xls' and pathlib.Path(dir_2).suffix == '.csv':
        print('file#1 is xls, file#2 is csv')
        df = pd.read_excel(dir_1)
        if delimiter_2 is not None:
            df2 = pd.read_csv(dir_2, delimiter=delimiter_2)
        else:
            df2 = pd.read_csv(dir_2)
    elif pathlib.Path(dir_1).suffix == '.xls' and pathlib.Path(dir_2).suffix == '.xls':
        print('both xls files, have style feature')
        df = pd.read_excel(dir_1)
        df2 = pd.read_excel(dir_2)
        book = xlrd.open_workbook(dir_1, formatting_info=True)
        book2 = xlrd.open_workbook(dir_2, formatting_info=True)
        has_style_feature = True
    else:
        print(pathlib.Path(dir_1).suffix, pathlib.Path(dir_2).suffix)
    print(df.info())
    print(df2.info())
    similarity_list = []
    # column header
    similarity_list.append(sim_header(get_header_set(df), get_header_set(df2)))
    # row header
    similarity_list.append(sim_row_header(df, df2))
    # cell type
    similarity_list.append(sim_tpye(get_cell_type_count(df), get_cell_type_count(df2)))
    # other style feature
    if has_style_feature is False:
        similarity_list.extend([0, 0, 0, 0, 0, 0, 0, 0])
    else:
        similarity_list.extend(sim_style_features(get_styles_feature(book), get_styles_feature(book2)))

    return similarity_list


def df_csv_sim_features(df_1, df_2):
    similarity_list = []
    # column header
    similarity_list.append(sim_header(get_header_set(df_1), get_header_set(df_2)))
    # row header
    similarity_list.append(sim_row_header(df_1, df_2))
    # cell type
    similarity_list.append(sim_tpye(get_cell_type_count(df_1), get_cell_type_count(df_2)))
    # other style feature
    has_style_feature = False
    if has_style_feature is False:
        similarity_list.extend([0, 0, 0, 0, 0, 0, 0, 0])
    else:
        similarity_list.extend(sim_style_features(get_styles_feature(book), get_styles_feature(book2)))

    return similarity_list


# input: a row in the dataset
# output: numeric, string, or mixture
def dataset_type(row):
    numeric, string = True, True
    for columnIndex, value in row.items():
        if (type(value) == int or type(value) == float or type(value).__module__ == 'numpy'):
            string = False
        if (type(value) == str):
            numeric = False
    if numeric is True:
        return 'numeric'
    elif string is True:
        return 'string'
    else:
        return 'mixture'


# input: two csv dataset in dataframe object, number of row/col to duplicate (default 1)
# output: a duplicated dataset
# case 1: same type (numeric / string), insert row from long row -> short row
# case 2: different type/mixture type: insert col from long column -> short column
def generate_duplicate_df(df_1, df_2, random_seed, frac_start=10, frac_end=30):
    random.seed(random_seed)
    # case 1
    df_1_type = dataset_type(df_1.iloc[0])
    df_2_type = dataset_type(df_2.iloc[0])
    dup_case = ''
    #     print(df_1_type,df_2_type)
    if df_1_type == df_2_type and df_1_type != 'mixture':
        #         print('case 1, dup row')
        dup_case = 'row'
        if len(df_1.columns) >= len(df_2.columns):
            dup_df, from_df = df_2.copy(deep=True), df_1
        else:
            dup_df, from_df = df_1.copy(deep=True), df_2
        ratio = random.randint(frac_start, frac_end) / 100.0  # set random ratio
        sample = from_df.sample(frac=ratio, random_state=42, replace=True)
        # cut from longer row to fit another table
        sample = sample.iloc[:, :len(dup_df.columns)]
        selected_rows = sample.values.tolist()
        for row in selected_rows:
            dup_df.loc[len(dup_df)] = row
    else:
        #         print('case 2, dup col')
        dup_case = 'col'
        if len(df_1) > len(df_2):
            dup_df = df_2.copy(deep=True)
            from_df = df_1.copy(deep=True)
        else:
            dup_df = df_1.copy(deep=True)
            from_df = df_2.copy(deep=True)
        ratio = random.randint(frac_start, frac_end) / 100.0  # set random ratio
        # cut from longer column to fit another table
        from_df_slice = from_df.head(len(dup_df))
        sample = from_df_slice.sample(axis='columns', frac=ratio, random_state=42, replace=True)
        # rename all column, otherwise column name will be bring to dup
        new_name_list = [id_generator() for i in range(len(sample.columns))]
        sample.columns = new_name_list
        sample.index = list(dup_df.index)
        dup_df = pd.concat([dup_df, sample], axis=1, ignore_index=True)
    #         print(dup)
    #         print(dup_df)
    return dup_df, from_df, dup_case, sample


# input: two column in list
# output: 0:different type 1: numeric<->numeric 2:string<-> string
def get_col_pair_type(col_1, col_2):
    if is_string_dtype(col_1) and is_string_dtype(col_2):
        return 2
    elif is_numeric_dtype(col_1) and is_numeric_dtype(col_2):
        return 1
    else:
        return 0


# initial implementation of value similarity - col
# input: two dataframe
# output: sum of similarity matrix
def intersec_col_sim(df_1, df_2):
    # shape of [col in df_1 * col in df_2]
    numeric_sim_metrix = np.zeros([df_1.shape[1], df_2.shape[1]])
    str_sim_metrix = np.zeros([df_1.shape[1], df_2.shape[1]])
    for col_1_index, column_1 in enumerate(df_1.columns[0:]):
        col_1 = df_1[column_1]
        col_1_numeric = pd.to_numeric(col_1, errors='coerce', downcast='float')
        col_1_numeric_set = set(col_1_numeric.tolist())
        col_1_str = col_1[col_1.apply(lambda x: isinstance(x, str))]
        col_1_str_set = set(col_1_str.tolist())
        for col_2_index, column_2 in enumerate(df_2.columns[0:]):
            col_2 = df_2[column_2]
            # 1. calculate  similarities of numeric value
            col_2_numeric = pd.to_numeric(col_2, errors='coerce', downcast='float')
            col_2_numeric_set = set(col_2_numeric.tolist())
            if len(col_1_numeric_set) != 0 or len(col_2_numeric_set) != 0:
                sim = len(col_1_numeric_set & col_2_numeric_set) / float(len(col_1_numeric_set | col_2_numeric_set))
                numeric_sim_metrix[col_1_index, col_2_index] = sim
            else:
                numeric_sim_metrix[col_1_index, col_2_index] = 0
            # 2. calculate  similarities of str value
            col_2_str = col_2[col_2.apply(lambda x: isinstance(x, str))]
            col_2_str_set = set(col_2_str.tolist())
            if len(col_1_str_set) != 0 or len(col_2_str_set) != 0:
                sim = len(col_1_str_set & col_2_str_set) / float(len(col_1_str_set | col_2_str_set))
                str_sim_metrix[col_1_index, col_2_index] = sim
            else:
                str_sim_metrix[col_1_index, col_2_index] = 0
    return np.sum(numeric_sim_metrix), np.sum(str_sim_metrix)


# initial implementation of value similarity - row
# input: two dataframe
# output: sum of similarity matrix
def intersec_row_sim(df_1, df_2):
    # shape of [col in df_1 * col in df_2]
    numeric_sim_metrix = np.zeros([df_1.shape[0], df_2.shape[0]])
    str_sim_metrix = np.zeros([df_1.shape[0], df_2.shape[0]])
    for row_1_index, row_1 in enumerate(df_1.itertuples()):
        row_1_numeric = pd.to_numeric(row_1, errors='coerce', downcast='float')
        row_1_numeric_set = set(row_1_numeric.tolist())
        row_1_str = [x for x in row_1 if isinstance(x, str)]
        row_1_str_set = set(row_1_str)
        for row_2_index, row_2 in enumerate(df_2.itertuples()):
            row_2_numeric = pd.to_numeric(row_2, errors='coerce', downcast='float')
            row_2_numeric_set = set(row_2_numeric.tolist())
            if (len(row_1_numeric_set) != 0 or len(row_2_numeric_set) != 0):
                sim = len(row_1_numeric_set & row_2_numeric_set) / float(len(row_1_numeric_set | row_2_numeric_set))
                numeric_sim_metrix[row_1_index, row_2_index] = sim
            else:
                numeric_sim_metrix[row_1_index, row_2_index] = 0
            row_2_str = [x for x in row_2 if isinstance(x, str)]
            row_2_str_set = set(row_2_str)
            if len(row_1_str_set) != 0 or len(row_2_str_set) != 0:
                sim = len(row_1_str_set & row_2_str_set) / float(len(row_1_str_set | row_2_str_set))
                str_sim_metrix[row_1_index, row_2_index] = sim
            else:
                str_sim_metrix[row_1_index, row_2_index] = 0
    return np.sum(numeric_sim_metrix), np.sum(str_sim_metrix)


# initial implementation of value similarity  with spicy distance
# input: two dataframe
# output: sum of similarity matrix
def spacy_sim(df_1, df_2, nlp):
    df_1_doc = ""
    df_2_doc = ""
    for col_1_index, column_1 in enumerate(df_1.columns[0:]):
        for col_2_index, column_2 in enumerate(df_2.columns[0:]):
            # 2. str
            col_1 = df_1[column_1]
            col_2 = df_2[column_2]
            col_1_str = col_1[col_1.apply(lambda x: isinstance(x, str))]
            col_2_str = col_2[col_2.apply(lambda x: isinstance(x, str))]
            col_1_str_list = col_1_str.tolist()
            col_2_str_list = col_2_str.tolist()
            df_1_doc += ' '.join(col_1_str_list)
            df_2_doc += ' '.join(col_2_str_list)
    doc1 = nlp(df_1_doc)
    doc2 = nlp(df_2_doc)

    return doc1.similarity(doc2)


# # textrank similarities
# def textrank_similarity(sents_1, sents_2):
#     counter = 0
#     for sent in sents_1:
#         if sent in sents_2:
#             counter += 1
#     sents_similarity=counter/(math.log(len(sents_1))+math.log(len(sents_2)))
#     return sents_similarity
# textrank similarities
def textrank_similarity(sents_1, sents_2):
    set_1 = set(sents_1)
    set_2 = set(sents_2)
    if (math.log(len(set_1)) + math.log(len(set_2))) == 0:
        return 0
    else:
        sents_similarity = len(set_1 & set_2) / (math.log(len(set_1)) + math.log(len(set_2)))
        return sents_similarity


def textrank_sim_all(df_1, df_2):
    text_df_1 = []
    text_df_2 = []
    for row_1_index, row_1 in enumerate(df_1.itertuples()):
        text_df_1.extend([x for x in row_1 if isinstance(x, str)])
    for row_2_index, row_2 in enumerate(df_2.itertuples()):
        text_df_2.extend([x for x in row_2 if isinstance(x, str)])
    if text_df_1 and text_df_2:
        return textrank_similarity(text_df_1, text_df_2)
    else:
        return 0


def tfidf_str_sim_all(df_1, df_2):
    tfidf_vectorizer = TfidfVectorizer()
    text_df_1 = []
    text_df_2 = []
    for row_1_index, row_1 in enumerate(df_1.itertuples()):
        text_df_1.extend([x for x in row_1 if isinstance(x, str)])
    for row_2_index, row_2 in enumerate(df_2.itertuples()):
        text_df_2.extend([x for x in row_2 if isinstance(x, str)])
    if text_df_1 and text_df_2:
        df_1_corpus = ' '.join(text_df_1)
        df_2_corpus = ' '.join(text_df_2)
        corpus = [df_1_corpus, df_2_corpus]
        try:
            mat = tfidf_vectorizer.fit_transform(corpus)
            return cosine_similarity(mat, mat)[0][1]
        except:
            return 0
    else:
        return 0


# initial implementation of value similarity - col
# input: two dataframe
# output: return similarity matrix as whole
# optimized for efficiency 6th-July
def intersec_col_sim_mat(df_1, df_2):
    numeric_sim_metrix = np.zeros([df_1.shape[1], df_2.shape[1]])
    str_sim_metrix = np.zeros([df_1.shape[1], df_2.shape[1]])
    col_1_str_set_list = []
    col_1_num_set_list = []
    col_2_str_set_list = []
    col_2_num_set_list = []
    columns_df_1 = list(df_1)
    columns_df_2 = list(df_2)
    for index_1, column_1 in enumerate(columns_df_1):
        col_1 = df_1[column_1]
        col_1_numeric = pd.to_numeric(col_1, errors='coerce', downcast='float')
        col_1_numeric_set = set(col_1_numeric.tolist())
        col_1_num_set_list.append(col_1_numeric_set)
        col_1_str = col_1[col_1.apply(lambda x: isinstance(x, str))]
        col_1_str_set = set(col_1_str.tolist())
        col_1_str_set_list.append(col_1_str_set)
    for index_2, column_2 in enumerate(columns_df_2):
        col_2 = df_2[column_2]
        col_2_numeric = pd.to_numeric(col_2, errors='coerce', downcast='float')
        col_2_numeric_set = set(col_2_numeric.tolist())
        col_2_num_set_list.append(col_2_numeric_set)
        col_2_str = col_2[col_2.apply(lambda x: isinstance(x, str))]
        col_2_str_set = set(col_2_str.tolist())
        col_2_str_set_list.append(col_2_str_set)
    for col_1_index, col_1_str_set in enumerate(col_1_str_set_list):
        if len(col_1_str_set) != 0:
            for col_2_index, col_2_str_set in enumerate(col_2_str_set_list):
                if len(col_2_str_set) != 0:
                    sim = len(col_1_str_set & col_2_str_set) / float(len(col_1_str_set | col_2_str_set))
                    str_sim_metrix[col_1_index, col_2_index] = sim
    for col_1_index, col_1_numeric_set in enumerate(col_1_num_set_list):
        if len(col_1_numeric_set) != 0:
            for col_2_index, col_2_numeric_set in enumerate(col_2_num_set_list):
                if len(col_2_numeric_set) != 0:
                    sim = len(col_1_numeric_set & col_2_numeric_set) / float(len(col_1_numeric_set | col_2_numeric_set))
                    numeric_sim_metrix[col_1_index, col_2_index] = sim

    return str_sim_metrix, numeric_sim_metrix


# initial implementation of value similarity - row
# input: two dataframe
# output: sum of similarity matrix
def intersec_row_sim_mat(df_1, df_2):
    # shape of [col in df_1 * col in df_2]
    numeric_sim_metrix = np.zeros([df_1.shape[0], df_2.shape[0]])
    str_sim_metrix = np.zeros([df_1.shape[0], df_2.shape[0]])
    row_1_str_set_list = []
    row_1_num_set_list = []
    row_2_str_set_list = []
    row_2_num_set_list = []
    for row_1_index, row_1 in enumerate(df_1.itertuples()):
        row_1_numeric = pd.to_numeric(row_1, errors='coerce', downcast='float')
        row_1_numeric_set = set(row_1_numeric.tolist())
        row_1_num_set_list.append(row_1_numeric_set)
        row_1_str = [x for x in row_1 if isinstance(x, str)]
        row_1_str_set = set(row_1_str)
        row_1_str_set_list.append(row_1_str_set)

    for row_2_index, row_2 in enumerate(df_2.itertuples()):
        row_2_numeric = pd.to_numeric(row_2, errors='coerce', downcast='float')
        row_2_numeric_set = set(row_2_numeric.tolist())
        row_2_num_set_list.append(row_2_numeric_set)
        row_2_str = [x for x in row_2 if isinstance(x, str)]
        row_2_str_set = set(row_2_str)
        row_2_str_set_list.append(row_2_str_set)

    for row_1_index, row_1_str_set in enumerate(row_1_str_set_list):
        if len(row_1_str_set) != 0:
            for row_2_index, row_2_str_set in enumerate(row_2_str_set_list):
                if len(row_2_str_set) != 0:
                    sim = len(row_1_str_set & row_2_str_set) / float(len(row_1_str_set | row_2_str_set))
                    str_sim_metrix[row_1_index, row_2_index] = sim
    for row_1_index, row_1_numeric_set in enumerate(row_1_num_set_list):
        if len(row_1_numeric_set) != 0:
            for row_2_index, row_2_numeric_set in enumerate(row_2_num_set_list):
                if len(row_2_numeric_set) != 0:
                    sim = len(row_1_numeric_set & row_2_numeric_set) / float(len(row_1_numeric_set | row_2_numeric_set))
                    numeric_sim_metrix[row_1_index, row_2_index] = sim

    return str_sim_metrix, numeric_sim_metrix


def get_features(s):
    width = 3
    s = s.lower()
    s = re.sub(r'[^\w]+', '', s)
    return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]


# input: two df
# output: two metrix, one for row-row simhash distance similarity, one for col-col simhash distance similarity
def simhash_distance_sim(df_1, df_2):
    df_1 = df_1.select_dtypes(include=[object])
    df_2 = df_2.select_dtypes(include=[object])
    if df_1.shape[1] < 1 or df_1.shape[1] < 1:
        return np.zeros((1, 1)), np.zeros((1, 1))

    # extract all string
    df_1_strs = [df_1[col_name].astype(str).tolist() for col_name in list(df_1)]
    df_2_strs = [df_2[col_name].astype(str).tolist() for col_name in list(df_2)]

    # column
    simhash_col_sim_mat = np.zeros([df_1.shape[1], df_2.shape[1]])
    df_1_cols_simhash = [Simhash(get_features(' '.join(i))) for i in df_1_strs]
    df_2_cols_simhash = [Simhash(get_features(' '.join(i))) for i in df_2_strs]
    for index_1, hash_1 in enumerate(df_1_cols_simhash):
        for index_2, hash_2 in enumerate(df_2_cols_simhash):
            simhash_col_sim_mat[index_1, index_2] = 1 - (hash_1.distance(hash_2) / 64)

    # row
    simhash_row_sim_mat = np.zeros([df_1.shape[0], df_2.shape[0]])
    rotate_df_1 = list(map(list, zip(*df_1_strs)))
    rotate_df_2 = list(map(list, zip(*df_2_strs)))
    df_1_rows_simhash = [Simhash(get_features(' '.join(i))) for i in rotate_df_1]
    df_2_rows_simhash = [Simhash(get_features(' '.join(i))) for i in rotate_df_2]
    for index_1, hash_1 in enumerate(df_1_rows_simhash):
        for index_2, hash_2 in enumerate(df_2_rows_simhash):
            simhash_row_sim_mat[index_1, index_2] = 1 - (hash_1.distance(hash_2) / 64)

    return simhash_col_sim_mat, simhash_row_sim_mat


# input: two df
# output: two metrix, one for row-row levenshtein distance similarity, one for col-col levenshtein distance similarity
def lev_distance_sim(df_1, df_2):
    df_1 = df_1.select_dtypes(include=[object])
    df_2 = df_2.select_dtypes(include=[object])
    if df_1.shape[1] < 1 or df_1.shape[1] < 1:
        return np.zeros((1, 1)), np.zeros((1, 1))

    # extract all string
    df_1_strs = [df_1[col_name].astype(str).tolist() for col_name in list(df_1)]
    df_2_strs = [df_2[col_name].astype(str).tolist() for col_name in list(df_2)]

    # column
    lev_col_sim_mat = np.zeros([df_1.shape[1], df_2.shape[1]])
    df_1_cols = [' '.join(i) for i in df_1_strs]
    df_2_cols = [' '.join(i) for i in df_2_strs]
    for index_1, str_1 in enumerate(df_1_cols):
        for index_2, str_2 in enumerate(df_2_cols):
            #             lev_col_sim_mat[index_1,index_2] = textdistance.levenshtein.normalized_similarity(str_1,str_2)
            lev_col_sim_mat[index_1, index_2] = Levenshtein.ratio(str_1, str_2)

    # row
    lev_row_sim_mat = np.zeros([df_1.shape[0], df_2.shape[0]])
    rotate_df_1 = list(map(list, zip(*df_1_strs)))
    rotate_df_2 = list(map(list, zip(*df_2_strs)))
    df_1_rows = [' '.join(i) for i in rotate_df_1]
    df_2_rows = [' '.join(i) for i in rotate_df_2]
    for index_1, str_1 in enumerate(df_1_rows):
        for index_2, str_2 in enumerate(df_2_rows):
            lev_row_sim_mat[index_1, index_2] = Levenshtein.ratio(str_1, str_2)

    return lev_col_sim_mat, lev_row_sim_mat


# input two df
# output:two metrix, one for row-row levenshtein distance similarity, one for col-col levenshtein distance similarity
def textrank_sim(df_1, df_2):
    df_1 = df_1.select_dtypes(include=[object])
    df_2 = df_2.select_dtypes(include=[object])
    if df_1.shape[1] < 1 or df_1.shape[1] < 1:
        return np.zeros((1, 1)), np.zeros((1, 1))

    # extract all string
    df_1_strs = [df_1[col_name].astype(str).tolist() for col_name in list(df_1)]
    df_2_strs = [df_2[col_name].astype(str).tolist() for col_name in list(df_2)]

    # column
    textrank_col_sim_mat = np.zeros([df_1.shape[1], df_2.shape[1]])
    df_1_cols = [' '.join(i) for i in df_1_strs]
    df_2_cols = [' '.join(i) for i in df_2_strs]
    for index_1, str_1 in enumerate(df_1_cols):
        for index_2, str_2 in enumerate(df_2_cols):
            textrank_col_sim_mat[index_1, index_2] = textrank_similarity(str_1, str_2)

    # row
    textrank_row_sim_mat = np.zeros([df_1.shape[0], df_2.shape[0]])
    rotate_df_1 = list(map(list, zip(*df_1_strs)))
    rotate_df_2 = list(map(list, zip(*df_2_strs)))
    df_1_rows = [' '.join(i) for i in rotate_df_1]
    df_2_rows = [' '.join(i) for i in rotate_df_2]
    for index_1, str_1 in enumerate(df_1_rows):
        for index_2, str_2 in enumerate(df_2_rows):
            textrank_row_sim_mat[index_1, index_2] = textrank_similarity(str_1, str_2)

    return textrank_col_sim_mat, textrank_row_sim_mat


# for numeric value similarity
def numeric_value_similarity(a, b):
    if a == 0 and b == 0:
        return 1
    else:
        return 1 - abs(a - b) / abs(a + b)


# for char level frequency similarity
def char_freq_similarity(line_a, line_b):
    res_1 = Counter(line_a.replace(" ", ""))
    res_2 = Counter(line_b.replace(" ", ""))
    diff = sum([abs(res_1[ele] - res_2[ele]) for ele in list(res_1 + res_2)])
    same = sum([abs(res_1[ele] + res_2[ele]) for ele in list(res_1 + res_2)])
    if same == 0:
        return 0
    else:
        return 1 - diff / same


# a rebuild value similarity function
# input: df_1, df_2
# output: sim_type,sim_mat_list
# sim_type:
#   0:str & num
#   1:str only
#   2:num only
#   3:one df is all str and the other is all num, so no clone
# sim_mat_list:
#   [jaccard_str_row_mat,jaccard_str_col_mat,
#   jaccard_num_row_mat,jaccard_num_col_mat,
#   mean_col_mat,dev_col_mat,
#   mean_row_mat,dev_row_mat,
#   simhash_col_mat,simhash_row_mat,
#   lev_col_mat,lev_row_mat,
#   textrank_col_mat,textrank_row_mat]
def val_sim(df_1, df_2):
    df_1_str = df_1.select_dtypes(include=[object])
    df_2_str = df_2.select_dtypes(include=[object])
    df_1_num = df_1.select_dtypes(include=[np.number])
    df_2_num = df_2.select_dtypes(include=[np.number])

    empty_str = False
    empty_num = False
    # we separate the table to str and numeric

    if df_1_str.shape[0] < 1 or df_2_str.shape[0] < 1 or df_1_str.shape[1] < 1 or df_2_str.shape[1] < 1:
        empty_str = True

    if df_1_num.shape[0] < 1 or df_2_num.shape[0] < 1 or df_1_num.shape[1] < 1 or df_2_num.shape[1] < 1:
        empty_num = True

    # part 1 jaccard index - str
    if empty_str:
        jaccard_str_row_mat = np.zeros((1, 1))
        jaccard_str_col_mat = np.zeros((1, 1))
    else:
        jaccard_str_row_mat = np.zeros([df_1_str.shape[0], df_2_str.shape[0]])
        jaccard_str_col_mat = np.zeros([df_1_str.shape[1], df_2_str.shape[1]])
        # jaccard str
        df_1_str_array = [df_1_str[col_name].astype(str).tolist() for col_name in list(df_1_str)]
        df_2_str_array = [df_2_str[col_name].astype(str).tolist() for col_name in list(df_2_str)]

        # jaccard str - col set
        df_1_str_array_col_set = [set(i) for i in df_1_str_array]
        df_2_str_array_col_set = [set(i) for i in df_2_str_array]
        # jaccard str - row set
        rotate_df_1_str_array = list(map(list, zip(*df_1_str_array)))
        rotate_df_2_str_array = list(map(list, zip(*df_2_str_array)))
        df_1_str_array_row_set = [set(i) for i in rotate_df_1_str_array]
        df_2_str_array_row_set = [set(i) for i in rotate_df_2_str_array]

        # jaccard str col matrix
        for index_1, df_1_str_col_set in enumerate(df_1_str_array_col_set):
            if len(df_1_str_col_set) != 0:
                for index_2, df_2_str_col_set in enumerate(df_2_str_array_col_set):
                    if len(df_2_str_col_set) != 0:
                        sim = len(df_1_str_col_set & df_2_str_col_set) / float(len(df_1_str_col_set | df_2_str_col_set))
                        jaccard_str_col_mat[index_1, index_2] = sim

        # jaccard str row matrix
        for index_1, df_1_str_row_set in enumerate(df_1_str_array_row_set):
            if len(df_1_str_row_set) != 0:
                for index_2, df_2_str_row_set in enumerate(df_2_str_array_row_set):
                    if len(df_2_str_row_set) != 0:
                        sim = len(df_1_str_row_set & df_2_str_row_set) / float(len(df_1_str_row_set | df_2_str_row_set))
                        jaccard_str_row_mat[index_1, index_2] = sim

                        # part 1 jaccard index - num
    if empty_num:
        jaccard_num_row_mat = np.zeros((1, 1))
        jaccard_num_col_mat = np.zeros((1, 1))
    else:
        jaccard_num_row_mat = np.zeros([df_1_num.shape[0], df_2_num.shape[0]])
        jaccard_num_col_mat = np.zeros([df_1_num.shape[1], df_2_num.shape[1]])
        # jaccard num
        df_1_num_array = [df_1_num[col_name].tolist() for col_name in list(df_1_num)]
        df_2_num_array = [df_2_num[col_name].tolist() for col_name in list(df_2_num)]

        # jaccard num - col set
        df_1_num_array_col_set = [set(i) for i in df_1_num_array]
        df_2_num_array_col_set = [set(i) for i in df_2_num_array]
        # jaccard num - row set
        rotate_df_1_num_array = list(map(list, zip(*df_1_num_array)))
        rotate_df_2_num_array = list(map(list, zip(*df_2_num_array)))
        df_1_num_array_row_set = [set(i) for i in rotate_df_1_num_array]
        df_2_num_array_row_set = [set(i) for i in rotate_df_2_num_array]

        # jaccard num col matrix
        for index_1, df_1_num_col_set in enumerate(df_1_num_array_col_set):
            if len(df_1_num_col_set) != 0:
                for index_2, df_2_num_col_set in enumerate(df_2_num_array_col_set):
                    if len(df_2_num_col_set) != 0:
                        sim = len(df_1_num_col_set & df_2_num_col_set) / float(len(df_1_num_col_set | df_2_num_col_set))
                        jaccard_num_col_mat[index_1, index_2] = sim

        # jaccard num row matrix
        for index_1, df_1_num_row_set in enumerate(df_1_num_array_row_set):
            if len(df_1_num_row_set) != 0:
                for index_2, df_2_num_row_set in enumerate(df_2_num_array_row_set):
                    if len(df_2_num_row_set) != 0:
                        sim = len(df_1_num_row_set & df_2_num_row_set) / float(len(df_1_num_row_set | df_2_num_row_set))
                        jaccard_num_row_mat[index_1, index_2] = sim

    # part 2 numeric_mean & numeric deviation
    if empty_num:
        mean_row_mat = np.zeros((1, 1))
        dev_row_mat = np.zeros((1, 1))
        mean_col_mat = np.zeros((1, 1))
        dev_col_mat = np.zeros((1, 1))
    else:
        df_1_num_np = df_1_num.to_numpy()
        df_2_num_np = df_2_num.to_numpy()
        mean_row_mat = np.zeros([df_1_num.shape[0], df_2_num.shape[0]])
        dev_row_mat = np.zeros([df_1_num.shape[0], df_2_num.shape[0]])

        # numeric row
        df_1_row_mean = np.nanmean(df_1_num_np, axis=1, dtype=np.float64)
        df_1_row_dev = np.nanstd(df_1_num_np, axis=1, dtype=np.float64)
        df_2_row_mean = np.nanmean(df_2_num_np, axis=1, dtype=np.float64)
        df_2_row_dev = np.nanstd(df_2_num_np, axis=1, dtype=np.float64)

        for index_1 in range(df_1_row_mean.shape[0]):
            for index_2 in range(df_2_row_mean.shape[0]):
                mean_row_mat[index_1][index_2] = numeric_value_similarity(df_1_row_mean[index_1],
                                                                          df_2_row_mean[index_2])
                dev_row_mat[index_1][index_2] = numeric_value_similarity(df_1_row_dev[index_1], df_2_row_dev[index_2])

        mean_col_mat = np.zeros([df_1_num.shape[1], df_2_num.shape[1]])
        dev_col_mat = np.zeros([df_1_num.shape[1], df_2_num.shape[1]])

        # numeric col
        df_1_col_mean = np.nanmean(df_1_num_np, axis=0, dtype=np.float64)
        df_1_col_dev = np.nanstd(df_1_num_np, axis=0, dtype=np.float64)
        df_2_col_mean = np.nanmean(df_2_num_np, axis=0, dtype=np.float64)
        df_2_col_dev = np.nanstd(df_2_num_np, axis=0, dtype=np.float64)

        for index_1 in range(df_1_col_mean.shape[0]):
            for index_2 in range(df_2_col_mean.shape[0]):
                mean_col_mat[index_1][index_2] = numeric_value_similarity(df_1_col_mean[index_1],
                                                                          df_2_col_mean[index_2])
                dev_col_mat[index_1][index_2] = numeric_value_similarity(df_1_col_dev[index_1], df_2_col_dev[index_2])
    # simhash row & simhash col
    # only for string
    from simhash import Simhash
    if empty_str:
        simhash_col_mat = np.zeros((1, 1))
        simhash_row_mat = np.zeros((1, 1))
    else:
        simhash_col_mat = np.zeros([df_1_str.shape[1], df_2_str.shape[1]])
        simhash_row_mat = np.zeros([df_1_str.shape[0], df_2_str.shape[0]])

        df_1_cols_simhash = [Simhash(get_features(' '.join(i))) for i in df_1_str_array]
        df_2_cols_simhash = [Simhash(get_features(' '.join(i))) for i in df_2_str_array]
        for index_1, hash_1 in enumerate(df_1_cols_simhash):
            for index_2, hash_2 in enumerate(df_2_cols_simhash):
                simhash_col_mat[index_1, index_2] = 1 - (hash_1.distance(hash_2) / 64)

        df_1_rows_simhash = [Simhash(get_features(' '.join(i))) for i in rotate_df_1_str_array]
        df_2_rows_simhash = [Simhash(get_features(' '.join(i))) for i in rotate_df_2_str_array]
        for index_1, hash_1 in enumerate(df_1_rows_simhash):
            for index_2, hash_2 in enumerate(df_2_rows_simhash):
                simhash_row_mat[index_1, index_2] = 1 - (hash_1.distance(hash_2) / 64)

    # levenshtein & textrank
    # only for string
    if empty_str:
        lev_col_mat = np.zeros((1, 1))
        lev_row_mat = np.zeros((1, 1))

        textrank_new_col_mat = np.zeros((1, 1))
        textrank_new_row_mat = np.zeros((1, 1))

    else:
        lev_col_mat = np.zeros([df_1_str.shape[1], df_2_str.shape[1]])
        lev_row_mat = np.zeros([df_1_str.shape[0], df_2_str.shape[0]])

        textrank_new_col_mat = np.zeros([df_1_str.shape[1], df_2_str.shape[1]])
        textrank_new_row_mat = np.zeros([df_1_str.shape[0], df_2_str.shape[0]])

        df_1_cols = [' '.join(i) for i in df_1_str_array]
        df_2_cols = [' '.join(i) for i in df_2_str_array]

        for index_1, str_1 in enumerate(df_1_cols):
            for index_2, str_2 in enumerate(df_2_cols):
                lev_col_mat[index_1, index_2] = Levenshtein.ratio(copy.deepcopy(str_1), copy.deepcopy(str_2))

                textrank_new_col_mat[index_1, index_2] = textrank_similarity(list(copy.deepcopy(str_1).split(' ')),
                                                                             list(copy.deepcopy(str_2).split(' ')))

        df_1_rows = [' '.join(i) for i in rotate_df_1_str_array]
        df_2_rows = [' '.join(i) for i in rotate_df_2_str_array]

        for index_1, str_1 in enumerate(df_1_rows):
            for index_2, str_2 in enumerate(df_2_rows):
                lev_row_mat[index_1, index_2] = Levenshtein.ratio(copy.deepcopy(str_1), copy.deepcopy(str_2))

                textrank_new_row_mat[index_1, index_2] = textrank_similarity(list(copy.deepcopy(str_1).split(' ')),
                                                                             list(copy.deepcopy(str_2).split(' ')))



    # a indicator value for helping model know which feature to focus
    if not empty_str and not empty_num:
        sim_type = 0
    elif empty_str and not empty_num:
        sim_type = 1
    elif not empty_str and empty_num:
        sim_type = 2
    elif empty_str and empty_num:
        sim_type = 3

    sim_mat_list = [jaccard_str_row_mat, jaccard_str_col_mat,
                    jaccard_num_row_mat, jaccard_num_col_mat,
                    mean_col_mat, dev_col_mat,
                    mean_row_mat, dev_row_mat,
                    simhash_col_mat, simhash_row_mat,
                    lev_col_mat, lev_row_mat,
                    textrank_new_col_mat, textrank_new_row_mat]

    return sim_type, sim_mat_list



