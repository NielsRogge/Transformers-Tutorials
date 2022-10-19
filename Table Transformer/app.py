import streamlit as st
from PIL import Image, ImageEnhance
import statistics
import os
import string
from collections import Counter 
from itertools import tee, count
# import TDTSR
import pytesseract
from pytesseract import Output
import json
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from cv2 import dnn_superres
from transformers import DetrFeatureExtractor
from transformers import DetrForObjectDetection
import torch
import asyncio
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout='wide')
st.title("Table Detection and Table Structure Recognition")
st.write("Implemented by MSFT team: https://github.com/microsoft/table-transformer")


def PIL_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) 

def cv_to_PIL(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


async def pytess(cell_pil_img):
    return ' '.join(pytesseract.image_to_data(cell_pil_img, output_type=Output.DICT, config='-c tessedit_char_blacklist=œ˜â€œï¬â™Ã©œ¢!|”?«“¥ --psm 6 preserve_interword_spaces')['text']).strip()


# def super_res(pil_img):
    # '''
    # Useful for low-res docs
    # '''
    # requires opencv-contrib-python installed without the opencv-python
    # sr = dnn_superres.DnnSuperResImpl_create()
    # image = PIL_to_cv(pil_img)
    # model_path = "/data/Salman/TRD/code/table-transformer/transformers/LapSRN_x2.pb"
    # model_name = 'lapsrn'
    # model_scale = 2
    # sr.readModel(model_path)
    # sr.setModel(model_name, model_scale)
    # final_img = sr.upsample(image)
    # final_img = cv_to_PIL(final_img)

    # return final_img


def sharpen_image(pil_img):

    img = PIL_to_cv(pil_img)
    sharpen_kernel = np.array([[-1, -1, -1], 
                               [-1,  9, -1], 
                               [-1, -1, -1]])

    sharpen = cv2.filter2D(img, -1, sharpen_kernel)
    pil_img = cv_to_PIL(sharpen)
    return pil_img


def uniquify(seq, suffs = count(1)):
    """Make all the items unique by adding a suffix (1, 2, etc).
    Credit: https://stackoverflow.com/questions/30650474/python-rename-duplicates-in-list-with-progressive-numbers-without-sorting-list
    `seq` is mutable sequence of strings.
    `suffs` is an optional alternative suffix iterable.
    """
    not_unique = [k for k,v in Counter(seq).items() if v>1] 

    suff_gens = dict(zip(not_unique, tee(suffs, len(not_unique))))  
    for idx,s in enumerate(seq):
        try:
            suffix = str(next(suff_gens[s]))
        except KeyError:
            continue
        else:
            seq[idx] += suffix

    return seq

def binarizeBlur_image(pil_img):
    image = PIL_to_cv(pil_img)
    thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)[1]

    result = cv2.GaussianBlur(thresh, (5,5), 0)
    result = 255 - result
    return cv_to_PIL(result)



def td_postprocess(pil_img):
    '''
    Removes gray background from tables
    '''
    img = PIL_to_cv(pil_img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 100), (255, 5, 255)) # (0, 0, 100), (255, 5, 255)
    nzmask = cv2.inRange(hsv, (0, 0, 5), (255, 255, 255)) # (0, 0, 5), (255, 255, 255))
    nzmask = cv2.erode(nzmask, np.ones((3,3))) # (3,3)
    mask = mask & nzmask

    new_img = img.copy()
    new_img[np.where(mask)] = 255


    return cv_to_PIL(new_img)

# def super_res(pil_img):
#     # requires opencv-contrib-python installed without the opencv-python
#     sr = dnn_superres.DnnSuperResImpl_create()
#     image = PIL_to_cv(pil_img)
#     model_path = "./LapSRN_x8.pb"
#     model_name = model_path.split('/')[1].split('_')[0].lower()
#     model_scale = int(model_path.split('/')[1].split('_')[1].split('.')[0][1])

#     sr.readModel(model_path)
#     sr.setModel(model_name, model_scale)
#     final_img = sr.upsample(image)
#     final_img = cv_to_PIL(final_img)

#     return final_img

def table_detector(image, THRESHOLD_PROBA):
    '''
    Table detection using DEtect-object TRansformer pre-trained on 1 million tables

    '''

    feature_extractor = DetrFeatureExtractor(do_resize=True, size=800, max_size=800)
    encoding = feature_extractor(image, return_tensors="pt")

    model = DetrForObjectDetection.from_pretrained("SalML/DETR-table-detection")

    with torch.no_grad():
        outputs = model(**encoding)

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD_PROBA

    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]

    return (model, probas[keep], bboxes_scaled)


def table_struct_recog(image, THRESHOLD_PROBA):
    '''
    Table structure recognition using DEtect-object TRansformer pre-trained on 1 million tables
    '''

    feature_extractor = DetrFeatureExtractor(do_resize=True, size=1000, max_size=1000)
    encoding = feature_extractor(image, return_tensors="pt")

    model = DetrForObjectDetection.from_pretrained("SalML/DETR-table-structure-recognition")
    with torch.no_grad():
        outputs = model(**encoding)

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD_PROBA

    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]

    return (model, probas[keep], bboxes_scaled)





class TableExtractionPipeline():

    colors = ["red", "blue", "green", "yellow", "orange", "violet"]

    # colors = ["red", "blue", "green", "red", "red", "red"]

    def add_padding(self, pil_img, top, right, bottom, left, color=(255,255,255)):
        '''
        Image padding as part of TSR pre-processing to prevent missing table edges
        '''
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result

    def plot_results_detection(self, c1, model, pil_img, prob, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
        '''
        crop_tables and plot_results_detection must have same co-ord shifts because 1 only plots the other one updates co-ordinates 
        '''
        # st.write('img_obj')
        # st.write(pil_img)
        plt.imshow(pil_img)
        ax = plt.gca()

        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            cl = p.argmax()
            xmin, ymin, xmax, ymax = xmin-delta_xmin, ymin-delta_ymin, xmax+delta_xmax, ymax+delta_ymax 
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,fill=False, color='red', linewidth=3))
            text = f'{model.config.id2label[cl.item()]}: {p[cl]:0.2f}'
            ax.text(xmin-20, ymin-50, text, fontsize=10,bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        c1.pyplot()


    def crop_tables(self, pil_img, prob, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
        '''
        crop_tables and plot_results_detection must have same co-ord shifts because 1 only plots the other one updates co-ordinates 
        '''
        cropped_img_list = []

        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):

            xmin, ymin, xmax, ymax = xmin-delta_xmin, ymin-delta_ymin, xmax+delta_xmax, ymax+delta_ymax 
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            cropped_img_list.append(cropped_img)


        return cropped_img_list

    def generate_structure(self, c2, model, pil_img, prob, boxes, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom):
        '''
        Co-ordinates are adjusted here by 3 'pixels'
        To plot table pillow image and the TSR bounding boxes on the table
        '''
        # st.write('img_obj')
        # st.write(pil_img)
        plt.figure(figsize=(32,20))
        plt.imshow(pil_img)
        ax = plt.gca()
        rows = {}
        cols = {}
        idx = 0


        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):

            xmin, ymin, xmax, ymax = xmin, ymin, xmax, ymax 
            cl = p.argmax()
            class_text = model.config.id2label[cl.item()]
            text = f'{class_text}: {p[cl]:0.2f}'
            # or (class_text == 'table column')
            if (class_text == 'table row')  or (class_text =='table projected row header') or (class_text == 'table column'):
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,fill=False, color=self.colors[cl.item()], linewidth=2))
                ax.text(xmin-10, ymin-10, text, fontsize=5, bbox=dict(facecolor='yellow', alpha=0.5))

            if class_text == 'table row':
                rows['table row.'+str(idx)] = (xmin, ymin-expand_rowcol_bbox_top, xmax, ymax+expand_rowcol_bbox_bottom)
            if class_text == 'table column':
                cols['table column.'+str(idx)] = (xmin, ymin-expand_rowcol_bbox_top, xmax, ymax+expand_rowcol_bbox_bottom)

            idx += 1


        plt.axis('on')
        c2.pyplot()
        return rows, cols

    def sort_table_featuresv2(self, rows:dict, cols:dict):
        # Sometimes the header and first row overlap, and we need the header bbox not to have first row's bbox inside the headers bbox
        rows_ = {table_feature : (xmin, ymin, xmax, ymax) for table_feature, (xmin, ymin, xmax, ymax) in sorted(rows.items(), key=lambda tup: tup[1][1])}
        cols_ = {table_feature : (xmin, ymin, xmax, ymax) for table_feature, (xmin, ymin, xmax, ymax) in sorted(cols.items(), key=lambda tup: tup[1][0])}

        return rows_, cols_

    def individual_table_featuresv2(self, pil_img, rows:dict, cols:dict):

        for k, v in rows.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            rows[k] = xmin, ymin, xmax, ymax, cropped_img

        for k, v in cols.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            cols[k] = xmin, ymin, xmax, ymax, cropped_img

        return rows, cols


    def object_to_cellsv2(self, master_row:dict, cols:dict, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom, padd_left):
        '''Removes redundant bbox for rows&columns and divides each row into cells from columns
        Args:

        Returns:

        
        '''
        cells_img = {}
        header_idx = 0
        row_idx = 0
        previous_xmax_col = 0
        new_cols = {}
        new_master_row = {}
        previous_ymin_row = 0
        new_cols = cols
        new_master_row = master_row
        ## Below 2 for loops remove redundant bounding boxes ###
        # for k_col, v_col in cols.items():
        #     xmin_col, _, xmax_col, _, col_img = v_col
        #     if (np.isclose(previous_xmax_col, xmax_col, atol=5)) or (xmin_col >= xmax_col):
        #         print('Found a column with double bbox')
        #         continue
        #     previous_xmax_col = xmax_col
        #     new_cols[k_col] = v_col

        # for k_row, v_row in master_row.items():
        #     _, ymin_row, _, ymax_row, row_img = v_row
        #     if (np.isclose(previous_ymin_row, ymin_row, atol=5)) or (ymin_row >= ymax_row):
        #         print('Found a row with double bbox')
        #         continue
        #     previous_ymin_row = ymin_row
        #     new_master_row[k_row] = v_row
        ######################################################
        for k_row, v_row in new_master_row.items():
            
            _, _, _, _, row_img = v_row
            xmax, ymax = row_img.size
            xa, ya, xb, yb = 0, 0, 0, ymax
            row_img_list = []
            # plt.imshow(row_img)
            # st.pyplot()
            for idx, kv in enumerate(new_cols.items()):
                k_col, v_col = kv
                xmin_col, _, xmax_col, _, col_img = v_col
                xmin_col, xmax_col = xmin_col - padd_left - 10, xmax_col - padd_left
                # plt.imshow(col_img)
                # st.pyplot()
                # xa + 3 : to remove borders on the left side of the cropped cell
                # yb = 3: to remove row information from the above row of the cropped cell
                # xb - 3: to remove borders on the right side of the cropped cell
                xa = xmin_col
                xb = xmax_col
                if idx == 0:
                    xa = 0
                if idx == len(new_cols)-1:
                    xb = xmax
                xa, ya, xb, yb = xa, ya, xb, yb

                row_img_cropped = row_img.crop((xa, ya, xb, yb))
                row_img_list.append(row_img_cropped)

            cells_img[k_row+'.'+str(row_idx)] = row_img_list
            row_idx += 1

        return cells_img, len(new_cols), len(new_master_row)-1

    def clean_dataframe(self, df):
        '''
        Remove irrelevant symbols that appear with tesseractOCR
        '''
        # df.columns = [col.replace('|', '') for col in df.columns]

        for col in df.columns:

            df[col]=df[col].str.replace("'", '', regex=True)
            df[col]=df[col].str.replace('"', '', regex=True)
            df[col]=df[col].str.replace(']', '', regex=True)
            df[col]=df[col].str.replace('[', '', regex=True)
            df[col]=df[col].str.replace('{', '', regex=True)
            df[col]=df[col].str.replace('}', '', regex=True)
        return df

    @st.cache
    def convert_df(self, df):
        return df.to_csv().encode('utf-8')


    def create_dataframe(self, c3, cells_pytess_result:list, max_cols:int, max_rows:int):
        '''Create dataframe using list of cell values of the table, also checks for valid header of dataframe
        Args:
            cells_pytess_result: list of strings, each element representing a cell in a table
            max_cols, max_rows: number of columns and rows
        Returns:
            dataframe : final dataframe after all pre-processing 
        '''

        headers = cells_pytess_result[:max_cols]
        new_headers = uniquify(headers, (f' {x!s}' for x in string.ascii_lowercase))
        counter = 0

        cells_list = cells_pytess_result[max_cols:]
        df = pd.DataFrame("", index=range(0, max_rows), columns=new_headers)

        cell_idx = 0
        for nrows in range(max_rows):
            for ncols in range(max_cols):
                df.iat[nrows, ncols] = str(cells_list[cell_idx])
                cell_idx += 1

        ## To check if there are duplicate headers if result of uniquify+col == col 
        ## This check removes headers when all headers are empty or if median of header word count is less than 6
        for x, col in zip(string.ascii_lowercase, new_headers):
            if f' {x!s}' == col:
                counter += 1
        header_char_count = [len(col) for col in new_headers]

        # if (counter == len(new_headers)) or (statistics.median(header_char_count) < 6):
        #     st.write('woooot')
        #     df.columns = uniquify(df.iloc[0], (f' {x!s}' for x in string.ascii_lowercase))
        #     df = df.iloc[1:,:]

        df = self.clean_dataframe(df)
        
        c3.dataframe(df)
        csv = self.convert_df(df)
        c3.download_button("Download table", csv, "file.csv", "text/csv", key='download-csv')
        
        return df






    async def start_process(self, image_path:str, TD_THRESHOLD, TSR_THRESHOLD, padd_top, padd_left, padd_bottom, padd_right, delta_xmin, delta_ymin, delta_xmax, delta_ymax, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom):
        '''
        Initiates process of generating pandas dataframes from raw pdf-page images

        '''
        image = Image.open(image_path).convert("RGB")
        model, probas, bboxes_scaled = table_detector(image, THRESHOLD_PROBA=TD_THRESHOLD)

        if bboxes_scaled.nelement() == 0:
            print('No table found in the pdf-page image'+image_path.split('/')[-1])
            return ''
        
        # try:
        # st.write('Document: '+image_path.split('/')[-1])
        c1, c2, c3 = st.columns((1,1,1))

        self.plot_results_detection(c1, model, image, probas, bboxes_scaled,  delta_xmin, delta_ymin, delta_xmax, delta_ymax) 
        cropped_img_list = self.crop_tables(image, probas, bboxes_scaled, delta_xmin, delta_ymin, delta_xmax, delta_ymax)

        for unpadded_table in cropped_img_list:

            table = self.add_padding(unpadded_table, padd_top, padd_right, padd_bottom, padd_left)
            # table = super_res(table)
            # table = binarizeBlur_image(table)
            # table = sharpen_image(table) # Test sharpen image next
            # table = td_postprocess(table)

            model, probas, bboxes_scaled = table_struct_recog(table, THRESHOLD_PROBA=TSR_THRESHOLD)
            rows, cols = self.generate_structure(c2, model, table, probas, bboxes_scaled, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom)
            # st.write(len(rows), len(cols))
            rows, cols = self.sort_table_featuresv2(rows, cols)
            master_row, cols = self.individual_table_featuresv2(table, rows, cols)

            cells_img, max_cols, max_rows = self.object_to_cellsv2(master_row, cols, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom, padd_left)

            sequential_cell_img_list = []
            for k, img_list in cells_img.items():
                for img in img_list:
                    # img = super_res(img)
                    # img = sharpen_image(img) # Test sharpen image next
                    # img = binarizeBlur_image(img)
                    # img = self.add_padding(img, 10,10,10,10)
                    # plt.imshow(img)
                    # c3.pyplot()
                    sequential_cell_img_list.append(pytess(img))

            cells_pytess_result = await asyncio.gather(*sequential_cell_img_list)
            

            self.create_dataframe(c3, cells_pytess_result, max_cols, max_rows)
            st.write('Errors in OCR is due to either quality of the image or performance of the OCR')
        # except:
        #     st.write('Either incorrectly identified table or no table, to debug remove try/except')
            # break
        # break




if __name__ == "__main__":

    img_name = st.file_uploader("Upload an image with table(s)")

    padd_top = st.slider('Padding top', 0, 200, 20)
    padd_left = st.slider('Padding left', 0, 200, 20)
    padd_right = st.slider('Padding right', 0, 200, 20)
    padd_bottom = st.slider('Padding bottom', 0, 200, 20)


    te = TableExtractionPipeline()
    # for img in image_list:
    if img_name is not None:
        asyncio.run(te.start_process(img_name, TD_THRESHOLD=0.6, TSR_THRESHOLD=0.8, padd_top=padd_top, padd_left=padd_left, padd_bottom=padd_bottom, padd_right=padd_right, delta_xmin=0, delta_ymin=0, delta_xmax=0, delta_ymax=0, expand_rowcol_bbox_top=0, expand_rowcol_bbox_bottom=0))



