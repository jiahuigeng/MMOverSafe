import pandas as pd
import os
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.utils.dataframe import dataframe_to_rows

# Specify the image directory
img_dir = "images"

def check_contains(text, ll):
    for item in ll:
        if item in text:
            return True
    return False

def get_topic(filename):
    for topic in ['violence', 'drugs', 'porngraphy', 'religion', 'illegal_activity', 'hate']:
        if topic in filename:
            return topic
    return None
def visualization(xlsx_file):

    # Read the original Excel file
    df = pd.read_excel(xlsx_file)
    img_dir = os.path.join("images", get_topic(xlsx_file))

    # Initialize a new DataFrame with required columns
    df_new = pd.DataFrame()
    df_new['Unsafe'] = df['Unsafe']
    df_new['Safe'] = df['Safe']
    df_new['Prompt'] = df['Prompt']

    # Add response columns and image columns in the specified order
    for j in range(5):
        df_new[f"unsafe_img_{j}"] = df[f"unsafe_img_{j}"]
        df_new[f"resp_hm_{j}"] = df[f"resp_hm_{j}"]

    for j in range(5):
        df_new[f"safe_img_{j}"] = df[f"safe_img_{j}"]
        df_new[f"safe_resp_hm_{j}"] = df[f"safe_resp_hm_{j}"]

    for i, row in df_new.iterrows():
        for j in range(5):
            content = row[f"resp_hm_{j}"]
            if not pd.isna(content):
                if not check_contains(content.lower(), ["sorry", "unable"]):
                    df_new.at[i, f"resp_hm_{j}"] = ''
                else:
                    print(content)

    # Create a new workbook and add rows from the DataFrame
    wb = Workbook()
    ws = wb.active

    # Write headers to the new workbook
    headers = list(df_new.columns)
    print(headers)
    ws.append(headers)

    # Write data rows
    for i, row in df_new.iterrows():
        ws.append(row.tolist())

    # Add images to the new workbook in the correct columns
    for i, row in enumerate(df.itertuples(), start=2):  # start=2 to skip header row
        for j in range(5):
            # row_series = pd.Series(row._asdict())
            # unsafe_resp = row_series[f"resp_hm_{j}"]
            # if not pd.isna(unsafe_resp):
            #     if not check_contains(unsafe_resp, ["unable", "sorry"]):
            #         # print(unsafe_resp)
            #         df.at[i, f"resp_hm_{j}"] = ''
            #     print(unsafe_resp)

            img_name = getattr(row, f"unsafe_img_{j}")
            if pd.isna(img_name): continue
            img_path = os.path.join(img_dir, img_name)
            # Check if the image file exists before inserting
            if os.path.exists(img_path):
                img = Image(img_path)
                img.width, img.height = 200, 200  # Resize if needed
                ws.add_image(img, f"{chr(68 + 2*j)}{i}")  # Columns start from 'D' for image insertion

            safe_img_name = getattr(row, f"safe_img_{j}")
            if pd.isna(safe_img_name): continue
            safe_img_path = os.path.join(img_dir, safe_img_name)
            if os.path.exists(safe_img_path):
                img = Image(safe_img_path)
                img.width, img.height = 200, 200  # Resize if needed
                ws.add_image(img, f"{chr(68+10 + 2*j)}{i}")

    # Save the new workbook
    save_xlsx_file = xlsx_file[:-5] + "_vis.xlsx"
    wb.save(save_xlsx_file)

    print(f"Visualization saved to {save_xlsx_file}")

def excel_column_name(n):
    name = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        name = chr(65 + remainder) + name
    return name

def visualization_10(xlsx_file):
    # Read the original Excel file
    df = pd.read_excel(xlsx_file)

    # Initialize a new DataFrame with required columns
    df_new = pd.DataFrame()
    df_new['Unsafe'] = df['Unsafe']
    df_new['Safe'] = df['Safe']
    df_new['Prompt'] = df['Prompt']
    topic = get_topic(xlsx_file)
    if topic == "drugs":
        img_dir = os.path.join("images", topic)
    else:
        img_dir = "images"
    # Add response columns and image columns in the specified order
    for j in range(5):
        df_new[f"unsafe_img_{j}"] = df[f"unsafe_img_{j}"]
        df_new[f"resp_hm_{j}"] = df[f"resp_hm_{j}"]

    for j in range(5):
        df_new[f"safe_img_{j}"] = df[f"safe_img_{j}"]
        df_new[f"safe_resp_hm_{j}"] = df[f"safe_resp_hm_{j}"]

    for j in range(5, 10):
        df_new[f"unsafe_img_{j}"] = df[f"unsafe_img_{j}"]
    for j in range(5, 10):
        df_new[f"safe_img_{j}"] = df[f"safe_img_{j}"]

    for i, row in df_new.iterrows():
        for j in range(5):
            content = row[f"resp_hm_{j}"]
            if not pd.isna(content):
                if not check_contains(content.lower(), ["sorry", "unable"]):
                    df_new.at[i, f"resp_hm_{j}"] = ''
                else:
                    print(content)

    # Create a new workbook and add rows from the DataFrame
    wb = Workbook()
    ws = wb.active

    # Write headers to the new workbook
    headers = list(df_new.columns)
    print(headers)
    ws.append(headers)

    # Write data rows
    for i, row in df_new.iterrows():
        ws.append(row.tolist())

    # Add images to the new workbook in the correct columns
    for i, row in enumerate(df.itertuples(), start=2):  # start=2 to skip header row
        for j in range(5):
            img_name = getattr(row, f"unsafe_img_{j}")
            if pd.isna(img_name): continue
            img_path = os.path.join(img_dir, img_name)
            # Check if the image file exists before inserting
            if os.path.exists(img_path):
                img = Image(img_path)
                img.width, img.height = 200, 200  # Resize if needed
                ws.add_image(img, f"{chr(68 + 2*j)}{i}")  # Columns start from 'D' for image insertion

            safe_img_name = getattr(row, f"safe_img_{j}")
            if pd.isna(safe_img_name): continue
            safe_img_path = os.path.join(img_dir, safe_img_name)
            if os.path.exists(safe_img_path):
                img = Image(safe_img_path)
                img.width, img.height = 200, 200  # Resize if needed
                ws.add_image(img, f"{chr(68+10 + 2*j)}{i}")

        for j in range(5,10):
            img_name = getattr(row, f"unsafe_img_{j}")
            if pd.isna(img_name): continue
            img_path = os.path.join(img_dir, img_name)
            # Check if the image file exists before inserting
            if os.path.exists(img_path):
                img = Image(img_path)
                img.width, img.height = 200, 200  # Resize if needed
                colname = excel_column_name(19 + j)
                ws.add_image(img, f"{colname}{i}")  # Columns start from 'D' for image insertion
        #
        for j in range(5, 10):
            safe_img_name = getattr(row, f"safe_img_{j}")
            if pd.isna(safe_img_name): continue
            safe_img_path = os.path.join(img_dir, safe_img_name)
            if os.path.exists(safe_img_path):
                img = Image(safe_img_path)
                img.width, img.height = 200, 200  # Resize if needed
                colname = excel_column_name(24 + j)
                ws.add_image(img, f"{colname}{i}")

    # Save the new workbook
    save_xlsx_file = xlsx_file[:-5] + "_vis10.xlsx"
    wb.save(save_xlsx_file)

if __name__ == "__main__":
    # visualization("0_drugs_update_hm_response.xlsx")
    # visualization_10("0_drugs_update_hm_response.xlsx")
    # visualization_10("0_violence_update_hm_response.xlsx")
    visualization_10("0_illegal_activity_update_hm_response.xlsx")
    visualization_10("0_religion_update_hm_response.xlsx")