from typing import List, Dict,Union, IO
import cv2, fitz ,re , os
import numpy as np
#from paddleocr import PaddleOCR
import pandas as pd





def pdf_to_images(pdf_input: Union[str, IO[bytes]]) -> List[np.ndarray]:
    """
    Converts a PDF to a list of OpenCV images.

    Args:
        pdf_input (str or BytesIO): File path or file-like object

    Returns:
        List of OpenCV (BGR) images, one per page
    """
    # For file objects, try to read as image first
    if not isinstance(pdf_input, str):
        try:
            pdf_input.seek(0)
            img_data = pdf_input.read()
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                return [img]
        except Exception:
            pass
        
        # If image reading failed, treat as PDF
        pdf_input.seek(0)
        doc = fitz.open(stream=pdf_input.read(), filetype="pdf")
    else:
        # For file paths, check if it's an image first
        if os.path.isfile(pdf_input):
            try:
                img = cv2.imread(pdf_input)
                if img is not None:
                    return [img]
            except Exception as e:
                print(f"Error reading image: {e}")
        
        # If not an image or reading failed, treat as PDF
        doc = fitz.open(pdf_input)

    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(2.0, 2.0)  # upscale for better OCR
        pix = page.get_pixmap(matrix=mat)
        image_data = pix.tobytes("png")

        # Convert image bytes to OpenCV format
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        images.append(img)

    doc.close()
    return images



# === Clean and normalize rows ===
def clean_table_rows(rows: List[List[str]]) -> List[List[str]]:
    cleaned = []
    for row in rows:
        if not any(cell.strip() for cell in row):
            continue
        cleaned_row = [re.sub(r'\s+', ' ', cell.strip()) for cell in row]
        cleaned.append(cleaned_row)
    return cleaned






# === Structure OCR outputs into rows ===
def structure_table_data(ocr_data: List[Dict]) -> List[List[str]]:
    if not ocr_data:
        return []

    sorted_data = sorted(ocr_data, key=lambda x: (x['y'], x['x']))
    
    # First pass: group by Y-coordinate
    rows = []
    current_row = []
    current_y = None
    y_threshold = 20

    for item in sorted_data:
        if current_y is None:
            current_y = item['y']
            current_row = [item]
        elif abs(item['y'] - current_y) <= y_threshold:
            current_row.append(item)
        else:
            row_texts = [i['text'] for i in sorted(current_row, key=lambda x: x['x'])]
            if any(row_texts):
                rows.append(row_texts)
            current_row = [item]
            current_y = item['y']

    if current_row:
        row_texts = [i['text'] for i in sorted(current_row, key=lambda x: x['x'])]
        if any(row_texts):
            rows.append(row_texts)

    # Second pass: merge continuation rows into table rows
    return merge_table_continuation_rows(clean_table_rows(rows))


def merge_table_continuation_rows(rows: List[List[str]]) -> List[List[str]]:
    """Merge continuation rows into proper table structure"""
    if not rows:
        return rows
    
    # Find header row pattern
    header_keywords = ['no', 'description', 'qty', 'um', 'price', 'worth', 'vat', 'gross']
    header_idx = -1
    
    for i, row in enumerate(rows):
        row_lower = [cell.lower() for cell in row]
        if len(row) >= 4 and sum(any(kw in cell for kw in header_keywords) for cell in row_lower) >= 3:
            header_idx = i
            break
    
    if header_idx == -1:
        return rows
    
    merged_rows = rows[:header_idx + 1]  # Keep everything up to and including header
    
    i = header_idx + 1
    while i < len(rows):
        current_row = rows[i]
        
        # Check if this looks like a main table row (starts with number)
        if (len(current_row) >= 3 and 
            current_row[0].strip() and 
            (current_row[0].strip().endswith('.') or current_row[0].strip().isdigit())):
            
            # This is a main row, check for continuation rows
            main_row = current_row[:]
            i += 1
            
            # Collect continuation rows (single column or short rows)
            continuation_texts = []
            while i < len(rows):
                next_row = rows[i]
                
                # Stop if we hit another main row or summary
                if (len(next_row) >= 3 and next_row[0].strip() and 
                    (next_row[0].strip().endswith('.') or next_row[0].strip().isdigit())):
                    break
                    
                # Stop if we hit summary keywords
                row_text = ' '.join(next_row).lower()
                if any(kw in row_text for kw in ['summary', 'total', 'subtotal', 'vat']):
                    break
                
                # This is a continuation row - add to description
                if (len(next_row) <= 2 and next_row[0].strip()) or \
                   (len(next_row) >= 3 and not next_row[0].strip() and next_row[1].strip()):
                    # Handle both single-column continuations and description-only rows
                    if next_row[0].strip():
                        continuation_texts.append(next_row[0].strip())
                    elif len(next_row) > 1 and next_row[1].strip():
                        continuation_texts.append(next_row[1].strip())
                    i += 1
                else:
                    break
            
            # Merge continuation text into description column (index 1)
            if continuation_texts and len(main_row) > 1:
                main_row[1] = main_row[1] + ' ' + ' '.join(continuation_texts)
            
            merged_rows.append(main_row)
        else:
            # Not a main table row, keep as is
            merged_rows.append(current_row)
            i += 1
    
    return merged_rows



# === Run PaddleOCR on each page image ===
def extract_table_data(pdf_path: str,ocr) -> List[List[str]]:
    images = pdf_to_images(pdf_path)

    # ocr = PaddleOCR(
    #     use_textline_orientation=True,
    #     lang='en'
    # )

    all_table_rows = []

    for i,img in enumerate(images):
        result = ocr.predict(img)

        if not result or not result[0]:
            continue

        ocr_dict = result[0]
        rec_texts = ocr_dict.get('rec_texts', [])
        rec_scores = ocr_dict.get('rec_scores', [])
        dt_polys = ocr_dict.get('dt_polys', [])

        ocr_data = []
        for text, score, poly in zip(rec_texts, rec_scores, dt_polys):
            if score < 0.6:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            ocr_data.append({
                'text': text.strip(),
                'x': center_x,
                'y': center_y,
                'confidence': score,
                'bbox': poly
            })

        page_table = structure_table_data(ocr_data)
        all_table_rows.extend(page_table)
        d = pd.DataFrame(all_table_rows)
        # print(f"\n iteration number : {i}\n")
        # print("extracted data", all_table_rows)
        # print('framed data\n',d)

    return all_table_rows



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def normalize(cell: str) -> str:
    """
    Clean the cell: lowercase, remove punctuation, extra spaces.
    """
    return re.sub(r'[^\w\s]', '', cell).strip().lower()


def extract_billing_table(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Extracts the billing table by identifying header row and filtering below it.
    """
    header_keywords = ['service description', 'description', 'product', 'item', 'qty', 'quantity', 'rate', 'price', 'amount', 'total']
    table_start_idx = -1

    # 1️⃣ Identify header row (must match at least 2 keywords)
    for i, row in df.iterrows():
        row_str = ' '.join([str(cell).lower() for cell in row if pd.notna(cell)])
        matches = sum(1 for kw in header_keywords if kw in row_str)
        #print(f"Row {i}: '{row_str}' - Matches: {matches}")
        if matches >= 2:
            table_start_idx = i
            #print(f"✅ Header found at row {i}: {list(row)}")
            break

    if table_start_idx == -1:
        print("❌ Billing table header not found.")
        return pd.DataFrame(), 0

    # 2️⃣ Extract all rows below the header
    df_table = df.iloc[table_start_idx:].reset_index(drop=True)

    # 3️⃣ Use first row as column header with proper handling
    if len(df_table) > 0:
        header_row = df_table.iloc[0]
        #print(f"Header row values: {list(header_row)}")
        #print(f"Null count in header: {header_row.isnull().sum()}")
        
        new_columns = []
        for i, col_val in enumerate(header_row):
            if pd.notna(col_val) and str(col_val).strip():
                clean_header = str(col_val).strip().replace('\n', ' ').replace('\r', '')
                new_columns.append(clean_header)
                #print(f"Column {i}: '{clean_header}'")
            else:
                new_columns.append(f'Column_{i}')
                #print(f"Column {i}: Empty, using 'Column_{i}'")
        
        # Set the column names
        #print(f"Setting columns to: {new_columns}")
        df_table.columns = new_columns
        # Remove the header row from data
        df_table = df_table.drop(0).reset_index(drop=True)
        
        # Remove any empty rows after header
        while len(df_table) > 0 and df_table.iloc[0].count() <= 1:
            df_table = df_table.drop(0).reset_index(drop=True)

    # 4️⃣ Stop scanning only if entire row looks like a summary (not just one column)
    stop_keywords = ['subtotal', 'tax', 'total', 'net', 'summary']
    stop_index = len(df_table)
    for i, row in df_table.iterrows():
        row_texts = [str(cell).lower() for cell in row if pd.notna(cell)]
        match_count = sum(any(kw in cell for kw in stop_keywords) for cell in row_texts)
       # match_count = sum(cell.strip().lower() in stop_keywords for cell in row_texts) 
        # ✅ Only stop if *all* cells in row contain stop keywords or row is sparse
        if match_count >= 2 or (len(row_texts) <= 2 and match_count >= 1):
            stop_index = i
            break


    # 5️⃣ Return only billing rows and actual end index
    df_billing = df_table.iloc[:stop_index].reset_index(drop=True)
   # print(f"Final billing table columns: {list(df_billing.columns)}")
    #print(f"Billing table shape: {df_billing.shape}")
    #print("df_billing\n", df_billing)
    
    # Calculate actual end index in original dataframe
    actual_end_index = table_start_idx + stop_index
    
    return df_billing, actual_end_index




def extract_summary_table_block(df: pd.DataFrame, billing_end_index: int) -> pd.DataFrame:
    import re

    # Expanded keywords to match real-world invoices
    summary_keywords = [
        'summary', 'subtotal', 'vat', 'gst', 'tax', 'total', 
        'invoice amount', 'net payable', 'amount due', 'withholding', 'grand total'
    ]

    summary_start = None
    n_rows = len(df)

    # Start scanning *after* the billing table
    for idx in range(billing_end_index + 1, n_rows):
        row = df.iloc[idx]
        row_str = " ".join(str(cell).lower() for cell in row if pd.notna(cell))

        # Exclude rows like "Tax ID", "Approved by", etc.
        if re.search(r'(tax\s*id|approved\s*by|finance department)', row_str):
            continue

        if any(kw in row_str for kw in summary_keywords):
            summary_start = idx
            break

    if summary_start is None:
        print("⚠️ Summary section not found.")
        return pd.DataFrame()

    # Collect rows until we hit a blank or footer-style row
    summary_rows = []
    for idx in range(summary_start, n_rows):
        row = df.iloc[idx]
        row_texts = [str(cell).strip().lower() for cell in row if pd.notna(cell)]

        if not row_texts:
            break  # Stop at empty row

        # Skip over footer blocks
        if any(re.search(r'(approved|authorized|prepared\s*by)', txt) for txt in row_texts):
            break

        summary_rows.append(row)

    if not summary_rows:
        print("⚠️ Summary rows were empty.")
        return pd.DataFrame()

    summary_df = pd.DataFrame(summary_rows).reset_index(drop=True)
    #print("summary df", summary_df)
    return summary_df






#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# === Main Pipeline ===
def process_invoice(pdf_path: str, output_csv: str = None):
    table_rows = extract_table_data(pdf_path)

    if not table_rows:
        print(" No table data found.")
        return

    df = pd.DataFrame(table_rows)

    if len(df) > 1 and all(df.iloc[0].notnull()):
        df.columns = df.iloc[0]
        df = df.drop(0).reset_index(drop=True)

    #print("\n Extracted Table:")
    #print(df)


    # --- Extract billing table only ---
    billing_df = extract_billing_table(df)

    if billing_df.empty:
        print(" Could not detect billing table.")
    else:
        print("\n Billing Table Detected:")
        #print(billing_df)

        # billing_csv = os.path.splitext(pdf_path)[0] + "_billing_table.csv"
        # billing_df.to_csv(billing_csv, index=False)
        # print(f"\n Billing Table saved to: {billing_csv}")


     # ✅ Extract summary info from remaining rows
    
    try:
        billing_end_index = df[df.eq(billing_df.iloc[-1]).all(axis=1)].index[-1]
    except IndexError:
        billing_end_index = df.shape[0] // 2  # Fallback to middle if not found
    
    
    summary_df = extract_summary_table_block(df,billing_end_index)


    if not summary_df.empty:
        #print("\n Summary Table Detected:")
        pass
        #print(summary_df)
        # summary_csv = os.path.splitext(pdf_path)[0] + "_summary_table.csv"
        # summary_df.to_csv(summary_csv, index=False)
        #print(f"\n Summary Table saved to: {summary_csv}")
    else:
        print(" No summary information found.")




def extract_invoice_info(df_input: Union[pd.DataFrame, List[List[str]]]) -> Dict[str, List[str]]:
    """
    Extracts organizational names and dates from rows above the billing table.
    """
    if isinstance(df_input, list):
        df = pd.DataFrame(df_input)
    elif isinstance(df_input, pd.DataFrame):
        df = df_input
    else:
        raise ValueError("Input must be a pandas DataFrame or a list of lists")

    header_keywords = ['description', 'product', 'item', 'qty', 'quantity', 'rate', 'price', 'amount', 'total']
    
    # 1️⃣ Locate billing table start
    table_start_idx = -1
    for i, row in df.iterrows():
        row_str = ' '.join([str(cell).lower() for cell in row if pd.notna(cell)])
        if sum(kw in row_str for kw in header_keywords) >= 2:
            table_start_idx = i
            break

    # 2️⃣ Get text lines above billing table
    header_rows = df.iloc[:table_start_idx] if table_start_idx != -1 else df
    text_lines = [' '.join([str(cell) for cell in row if pd.notna(cell)]).strip()
                  for row in header_rows.values]

    # 3️⃣ Define organization and date patterns
    org_keywords = ['Client:', 'Vendor:', 'Billed By', 'Billed To']
    date_pattern = re.compile(
        r'\b(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})\b',
        re.IGNORECASE
    )

    organizations = set()
    dates = set()


    for line in text_lines:
        lower_line = line.lower()
        if (line and
            len(line.split()) >= 2 and
            not any(char.isdigit() for char in line) and
            '@' not in line and
            not any(term in lower_line for term in ['invoice', 'date', 'project', 'payment', 'summary', 'ref'])):
            organizations.add(line.strip())

        # Extract dates
        matches = date_pattern.findall(line)
        for match in matches:
            try:
                dt = pd.to_datetime(match)
                dates.add(dt.strftime('%Y-%m-%d'))
            except Exception:
                pass

    # 4️⃣ Clean organization list
    filter_words = {'client', 'vendor', 'record', 'invoice', 'ref', 'status', 'approved', 'by'}
    organizations = [org for org in organizations if org.lower() not in filter_words and len(org.split()) > 1]

    return {
        'organizations': list(organizations),
        'dates': list(dates)
    }



