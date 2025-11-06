# views.py

import os
import tempfile
import pandas as pd
import json
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from .extractor import extract_table_data, extract_invoice_info, extract_billing_table, extract_summary_table_block
#from .comparator import find_closest_product
from fuzzywuzzy import fuzz, process


from paddleocr import PaddleOCR
from django.views.decorators.csrf import csrf_exempt

ocr = PaddleOCR(
         use_textline_orientation=True,
         lang='en'
     )




def home(request):
    #print("home request hit ")

    return render(request, 'invoice/home.html')

@csrf_exempt
def upload_invoices(request):
    #print("upload_invoice got hit")
    if request.method == 'POST':
        file1 = request.FILES.get('invoice1')
        file2 = request.FILES.get('invoice2')

        if not file1 or not file2:
            return JsonResponse({'error': 'Both invoices are required'}, status=400)

        def extract_data(file):
            
            try:
                table_rows = extract_table_data(file, ocr)
                if not table_rows:
                    return {}, [], []
            except Exception as e:
                print(f"Error in extract_table_data: {e}")
                return {}, [], []
            df = pd.DataFrame(table_rows)
            #print(f"Initial DataFrame columns: {list(df.columns)}")
            #print(f"DataFrame shape: {df.shape}")
            
            info = extract_invoice_info(table_rows)
            # Optional header row detection
            # if len(df) > 1 and df.iloc[0].notnull().sum() >= df.shape[1] - 2:
            #     df.columns = df.iloc[0]
            #     df = df.drop(0).reset_index(drop=True)
            billing_df, billing_end_index = extract_billing_table(df)
            summary_df = extract_summary_table_block(df, billing_end_index)
            
            #print(f"Billing DataFrame columns after extraction: {list(billing_df.columns)}")
            #print(f"Summary DataFrame columns after extraction: {list(summary_df.columns)}")
            
            # fill NaNs or None so JSON is cleaner
            billing_json = billing_df.fillna('').to_dict(orient="records")
            summary_json = summary_df.fillna('').to_dict(orient="records")
            return info, billing_json, summary_json

        info1, billing1, summary1 = extract_data(file1)
        info2, billing2, summary2 = extract_data(file2)

        # Store data to session_data for later match
        request.session['info1'] = info1
        request.session['info2'] = info2
        request.session['billing1'] = billing1
        request.session['billing2'] = billing2
        request.session['summary1'] = summary1
        request.session['summary2'] = summary2

        # Return JSON
        return JsonResponse({
            'billing1': billing1,
            'billing2': billing2,
            'summary1': summary1,
            'summary2': summary2,
        }, status=200)

    # If GET or no files — bad request
    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def match_product_view(request):
    if request.method == "POST":
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON body'}, status=400)

        clicked_value = body.get("value", "").strip().lower()
        source_table = body.get("source", "")  # "billing1" or "billing2"

        billing1 = request.session.get("billing1", [])
        billing2 = request.session.get("billing2", [])
        summary1 = request.session.get("summary1", [])
        summary2 = request.session.get("summary2", [])



        # Decide which table to search
        #search_table = billing2 if source_table == "billing1" else billing1
        if source_table == "billing1":
            search_table = billing2
        elif source_table == "billing2":
            search_table = billing1
        elif source_table == "summary1":
            search_table = summary2
        elif source_table == "summary2":
            search_table = summary1
        else:
            search_table = []       

        if not clicked_value or not search_table:
            return JsonResponse({"matched_index": -1})

        df = pd.DataFrame(search_table)
        # Flatten all cell values to a list of (row_idx, col_idx, value)
        cell_list = []
        for row_idx, row in df.iterrows():
            for col in df.columns:
                val = str(row[col]).strip().lower()
                cell_list.append((row_idx, col, val))

        # Fuzzy match clicked_value to all cell values
        values = [cell[2] for cell in cell_list]
        match, score = process.extractOne(clicked_value, values, scorer=fuzz.token_sort_ratio)
        if score >= 70:
            # Find the row index of the matched cell
            for row_idx, col, val in cell_list:
                if val == match:
                    return JsonResponse({"matched_index": row_idx})
        return JsonResponse({"matched_index": -1})

    return JsonResponse({"error": "Invalid request"}, status=400)


@csrf_exempt
def diff_identifier(request):

    if request.method == "POST":
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON body'}, status=400)

        clicked_value = body.get("value", "").strip().lower()
        source_table = body.get("source", "")  # "billing1" or "billing2"
        matched_index = body.get("matched_index", None)

        billing1 = request.session.get("billing1", [])
        billing2 = request.session.get("billing2", [])

        source_data = billing1 if source_table == "billing1" else billing2
        target_data = billing2 if source_table == "billing1" else billing1

        if not clicked_value or not source_data or not target_data:
            return JsonResponse({"matched_index": -1})

        source_df = pd.DataFrame(source_data)
        target_df = pd.DataFrame(target_data)

        # Find the row index in source_df containing the clicked value
        clicked_row_idx = None
        for idx, row in source_df.iterrows():
            if any(str(cell).strip().lower() == clicked_value for cell in row):
                clicked_row_idx = idx
                break

        if clicked_row_idx is None or matched_index is None or matched_index < 0 or matched_index >= len(target_df):
            return JsonResponse({"matched_index": -1})

        clicked_row = source_df.iloc[clicked_row_idx]
        matched_row = target_df.iloc[matched_index].to_dict()

        # Identify mismatched columns
        common_cols = set(source_df.columns) & set(target_df.columns)
        mismatches = []
        for col in common_cols:
            if str(clicked_row[col]) != str(matched_row.get(col, "")):
                mismatches.append(col)

        return JsonResponse({
            "matched_index": matched_index,
            "matched_row": matched_row,
            "clicked_row": clicked_row.to_dict(),
            "mismatches": mismatches,
        })

    return JsonResponse({"error": "Invalid request"}, status=400)



from collections import defaultdict

@csrf_exempt
def row_comparator(request):
    if request.method == "POST":
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON body'}, status=400)

        selected_column = body.get("column", "").strip().lower()

        selected_key = body.get("key", "").strip().lower()
        if not selected_column:
            return JsonResponse({'error': 'No column selected'}, status=400)

        billing1 = request.session.get("billing1", [])
        billing2 = request.session.get("billing2", [])

        if not billing1 or not billing2:
            return JsonResponse({'error': 'No billing data'}, status=400)

        df1 = pd.DataFrame(billing1)
        df2 = pd.DataFrame(billing2)

        # print("Selected column:", selected_column)
        # print("df1 columns:", df1.columns)
        # print("df2 columns:", df2.columns)

        # Fuzzy match column names (ignore case, ≥80% match)
        def best_match(col, columns):
            matches = process.extract(col, columns, scorer=fuzz.token_sort_ratio)
            for match, score in matches:
                if score >= 80:
                    return match
            return None

        col1 = best_match(selected_column, [c.lower() for c in df1.columns])
        col2 = best_match(selected_column, [c.lower() for c in df2.columns])

        if not col1 or not col2:
            return JsonResponse({'error': 'Column not found in one or both invoices'}, status=400)

        orig_col1 = [c for c in df1.columns if c.lower() == col1][0]
        orig_col2 = [c for c in df2.columns if c.lower() == col2][0]



        if selected_key:
    # Fuzzy match all rows for selected product
            def match_rows(df, col, key):
                matches = []
                for idx, row in df.iterrows():
                    val = str(row[col]).strip()
                    if fuzz.token_sort_ratio(val.lower(), key.lower()) >= 80:
                        matches.append(row)
                return matches

            rows1 = match_rows(df1, orig_col1, selected_key)
            rows2 = match_rows(df2, orig_col2, selected_key)

            all_cols = set(df1.columns) | set(df2.columns)
            result = []

            # Compare each pair (if repeated, compare by index; if more in one, show empty for missing)
            max_len = max(len(rows1), len(rows2))
            for i in range(max_len):
                row1 = rows1[i] if i < len(rows1) else pd.Series()
                row2 = rows2[i] if i < len(rows2) else pd.Series()
                invoice1 = {}
                invoice2 = {}
                diffs = {}
                for col in all_cols:
                    val1 = str(row1[col]) if col in row1 else ""
                    val2 = str(row2[col]) if col in row2 else ""
                    invoice1[col] = val1
                    invoice2[col] = val2
                    diffs[col] = (val1 != val2)
                result.append({
                    "key": selected_key,
                    "invoice1": invoice1,
                    "invoice2": invoice2,
                    "diffs": diffs
                })

            return JsonResponse({
                "key": selected_key,
                "all_columns": list(all_cols),
                "rows": result
            })





        # Build mapping by product name (or selected column value)
        products1 = {str(row[orig_col1]).strip(): row for _, row in df1.iterrows()}
        products2 = {str(row[orig_col2]).strip(): row for _, row in df2.iterrows()}

        all_keys = set(products1.keys()) | set(products2.keys())
        result = []

        for key in all_keys:
            row1 = products1.get(key, {})
            row2 = products2.get(key, {})
            row_result = {
                "key": key,
                "invoice1": {},
                "invoice2": {},
                "diffs": {}
            }
            # Compare all columns present in either invoice
            all_cols = set(df1.columns) | set(df2.columns)
            for col in all_cols:
                # For pandas Series, use .get(col, "") or row[col] if col in row
                if isinstance(row1, pd.Series):
                    val1 = str(row1[col]) if col in row1 else ""
                elif isinstance(row1, dict):
                    val1 = str(row1.get(col, ""))
                else:
                    val1 = ""
                if isinstance(row2, pd.Series):
                    val2 = str(row2[col]) if col in row2 else ""
                elif isinstance(row2, dict):
                    val2 = str(row2.get(col, ""))
                else:
                    val2 = ""
                row_result["invoice1"][col] = val1
                row_result["invoice2"][col] = val2
                row_result["diffs"][col] = (val1 != val2)
            result.append(row_result)

        return JsonResponse({
            "key_column_invoice1": orig_col1,
            "key_column_invoice2": orig_col2,
            "all_columns": list(set(df1.columns) | set(df2.columns)),
            "rows": result
        })

    return JsonResponse({'error': 'Invalid request'}, status=400)



import re

def normalize_name(name):
    # Lowercase, remove spaces and punctuation for grouping
    return re.sub(r'[\s\-]+', '', name.lower())


@csrf_exempt
def get_product_keys(request):
    if request.method == "GET":
        billing1 = request.session.get("billing1", [])
        billing2 = request.session.get("billing2", [])

        if not billing1 or not billing2:
            return JsonResponse({'error': 'No billing data'}, status=400)

        df1 = pd.DataFrame(billing1)
        df2 = pd.DataFrame(billing2)

        # Candidate column names
        candidates = ["product", "description", "name", "item", "service"]

        def best_match(columns):
            matches = process.extractOne(
                "product", [c.lower() for c in columns], scorer=fuzz.token_sort_ratio
            )
            if matches and matches[1] >= 80:
                return [c for c in columns if c.lower() == matches[0]][0]
            for cand in candidates:
                matches = process.extractOne(
                    cand, [c.lower() for c in columns], scorer=fuzz.token_sort_ratio
                )
                if matches and matches[1] >= 80:
                    return [c for c in columns if c.lower() == matches[0]][0]
            return columns[0]  # fallback

        col1 = best_match(df1.columns)
        col2 = best_match(df2.columns)

        products1 = df1[col1].dropna().astype(str).str.strip().tolist() if col1 in df1 else []
        products2 = df2[col2].dropna().astype(str).str.strip().tolist() if col2 in df2 else []

        # Normalize and group products
        all_products = products1 + products2
        groups = {}
        for prod in all_products:
            norm = normalize_name(prod)
            # Fuzzy group: if norm matches any existing group, use that
            if norm in groups:
             groups[norm].append(prod)
            else:
                groups[norm] = [prod]

        # For dropdown, use the first representative name from each group
        normalized = [group[0] for group in groups.values()]
        normalized = sorted(normalized, key=lambda x: x.lower())

        return JsonResponse({
            "product_column_invoice1": col1,
            "product_column_invoice2": col2,
            "products": normalized
        })

    return JsonResponse({'error': 'Invalid request'}, status=400)



COLUMN_MAPPING = {
    "description": ["description", "product", "item", "service", "product name"],
    "qty": ["qty", "quantity", "count"],
    "unit": ["unit", "unit price", "price per unit", "rate"],
    "total": ["total", "grand total","Total (INR)","total (USD)" ,"amount", "subtotal"],
    "amount": ["Amount", "Amount inr", "Amount usd", "Amount (INR)", "Amount (USD)"],  # Added for handling currencies
    "name": ["name", "organization", "company"],  # Added for organizations
    "date": ["date", "invoice date", "issue date"]  # Added for dates
}

@csrf_exempt
def column_comparator(request):
    if request.method == "POST":
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON body'}, status=400)

        selected_column = body.get("column", "").strip().lower()

        if not selected_column:
            return JsonResponse({'error': 'No column selected'}, status=400)

        billing1 = request.session.get("billing1", [])
        billing2 = request.session.get("billing2", [])
        info1 = request.session.get("info1", [])
        info2 = request.session.get("info2", [])

        if not billing1 or not billing2:
            return JsonResponse({'error': 'No billing data'}, status=400)

        # Ensure info1 and info2 are parsed as JSON objects (if they are JSON strings)
        if isinstance(info1, str):
            try:
                info1 = json.loads(info1)  # Parse JSON string into a dictionary/list
            except json.JSONDecodeError:
                return JsonResponse({'error': 'Invalid data in info1'}, status=400)

        if isinstance(info2, str):
            try:
                info2 = json.loads(info2)  # Parse JSON string into a dictionary/list
            except json.JSONDecodeError:
                return JsonResponse({'error': 'Invalid data in info2'}, status=400)

        # If column is 'name' or 'date', return data from info1 or info2
        if selected_column == "name":
            return JsonResponse({
                "selected_column": selected_column,
                "organization1": info1.get("organizations", []),
                "organization2": info2.get("organizations", [])
            })

        if selected_column == "date":
            return JsonResponse({
                "selected_column": selected_column,
                "date1": info1.get("dates", []),
                "date2": info2.get("dates", [])
            })

        # For other columns, process the billing data as before
        df1 = pd.DataFrame(billing1)
        df2 = pd.DataFrame(billing2)

        # Function to normalize column names by removing currency suffixes
        def normalize_column_name(col_name):
            for suffix in [" (INR)", " (USD)", "(INR)", "(USD)"]:
                if col_name.endswith(suffix):
                    col_name = col_name.replace(suffix, "")
            return col_name.strip().lower()

        # Function to match the column name using COLUMN_MAPPING
        def find_column(column_name, df_columns):
            # Clean columns (strip any leading/trailing spaces and normalize)
            cleaned_columns = [normalize_column_name(col).lower() for col in df_columns]

            # First, check for an exact match with the selected column
            if column_name in cleaned_columns:
                exact_match = df_columns[cleaned_columns.index(column_name)]
                #print(f"Debug: Exact match found: {exact_match}")
                return exact_match

            # If no exact match, try checking synonyms from COLUMN_MAPPING
            possible_columns = COLUMN_MAPPING.get(column_name, [])
            for possible_column in possible_columns:
                normalized_possible_column = normalize_column_name(possible_column)
                if normalized_possible_column in cleaned_columns:
                    match = df_columns[cleaned_columns.index(normalized_possible_column)]
                    #print(f"Debug: Found column synonym match: {match}")
                    return match

            # If no exact match or synonym match, try fuzzy matching from the COLUMN_MAPPING
            if possible_columns:
                matches = process.extract(column_name, possible_columns, scorer=fuzz.token_sort_ratio)

                # Log matches for debugging
                #print(f"Debug: Matching '{column_name}' to possible columns: {possible_columns}")
                #print(f"Debug: Matches: {matches}")

                for match, score in matches:
                    if score >= 70:  # Adjusted threshold to 70
                        for col in df_columns:
                            if fuzz.token_sort_ratio(col.lower(), match.lower()) >= 70:
                                #print(f"Debug: Found column match '{match}' -> '{col}'")
                                return col

            print(f"Debug: No column match found for '{column_name}'")
            return None

        # Find the columns in both df1 and df2
        col1 = find_column(selected_column, df1.columns)
        col2 = find_column(selected_column, df2.columns)

        if not col1 or not col2:
            return JsonResponse({'error': 'Column not found in one or both invoices'}, status=400)

        # Compare the columns side-by-side
        col1_values = df1[col1].fillna('')
        col2_values = df2[col2].fillna('')

        max_len = max(len(col1_values), len(col2_values))

        col1_values = col1_values.tolist() + [''] * (max_len - len(col1_values))
        col2_values = col2_values.tolist() + [''] * (max_len - len(col2_values))

        result = []
        for i in range(max_len):
            result.append({
                "invoice1": col1_values[i],
                "invoice2": col2_values[i],
                "diff": col1_values[i] != col2_values[i]
            })

        return JsonResponse({
            "selected_column": selected_column,
            "invoice1_column": col1,
            "invoice2_column": col2,
            "comparison_result": result
        })

    return JsonResponse({'error': 'Invalid request'}, status=400)



@csrf_exempt
def get_dropdown_columns(request):
    """API to get column names from both invoices for dropdown selection"""
    if request.method == "GET":
        billing1 = request.session.get("billing1", [])
        billing2 = request.session.get("billing2", [])
        
        if not billing1 or not billing2:
            return JsonResponse({'error': 'No billing data found'}, status=400)
        
        df1 = pd.DataFrame(billing1)
        df2 = pd.DataFrame(billing2)
        
        return JsonResponse({
            'invoice1_columns': list(df1.columns),
            'invoice2_columns': list(df2.columns)
        })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def compare_selected_columns(request):
    """API to compare selected columns from both invoices side by side"""
    if request.method == "POST":
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON body'}, status=400)
        
        # Handle both single column comparison and multiple combinations
        if 'column_combinations' in body:
            # Multiple combinations
            column_combinations = body.get('column_combinations', [])
            
            if not column_combinations:
                return JsonResponse({'error': 'No column combinations provided'}, status=400)
            
            billing1 = request.session.get("billing1", [])
            billing2 = request.session.get("billing2", [])
            
            if not billing1 or not billing2:
                return JsonResponse({'error': 'No billing data found'}, status=400)
            
            df1 = pd.DataFrame(billing1)
            df2 = pd.DataFrame(billing2)
            
            # Build combined columns maintaining row relationships
            combined_columns = []
            
            for combo in column_combinations:
                col1 = combo.get('invoice1_column')
                col2 = combo.get('invoice2_column')
                combined_name = combo.get('combined_name', f'{col1} + {col2}')
                
                if col1 not in df1.columns or col2 not in df2.columns:
                    continue
                
                # Get values from both columns
                col1_values = df1[col1].fillna('').astype(str).tolist()
                col2_values = df2[col2].fillna('').astype(str).tolist()
                
                # Perform fuzzy mapping between the two columns
                paired_values = []
                used_col2_indices = set()
                
                # Map each value from col1 to best match in col2
                for val1 in col1_values:
                    if not val1.strip():
                        paired_values.append({'col1': val1, 'col2': ''})
                        continue
                    
                    best_match = ''
                    best_score = 0
                    best_idx = -1
                    
                    for i, val2 in enumerate(col2_values):
                        if i in used_col2_indices or not val2.strip():
                            continue
                        
                        score = fuzz.token_sort_ratio(val1.lower(), val2.lower())
                        if score > best_score and score >= 70:
                            best_match = val2
                            best_score = score
                            best_idx = i
                    
                    if best_idx != -1:
                        used_col2_indices.add(best_idx)
                        paired_values.append({'col1': val1, 'col2': best_match})
                    else:
                        paired_values.append({'col1': val1, 'col2': ''})
                
                # Add unmatched values from col2
                for i, val2 in enumerate(col2_values):
                    if i not in used_col2_indices and val2.strip():
                        paired_values.append({'col1': '', 'col2': val2})
                
                combined_columns.append({
                    'name': combined_name,
                    'paired_values': paired_values
                })
            
            # Find maximum length across all paired columns
            max_len = max(len(col['paired_values']) for col in combined_columns) if combined_columns else 0
            
            # Pad columns to same length
            for col in combined_columns:
                while len(col['paired_values']) < max_len:
                    col['paired_values'].append({'col1': '', 'col2': ''})
            
            # Build table data with paired structure
            table_data = []
            headers = []
            
            for col in combined_columns:
                headers.extend([f"{col['name']} (Invoice 1)", f"{col['name']} (Invoice 2)"])
            
            for i in range(max_len):
                row = {}
                for col in combined_columns:
                    pair = col['paired_values'][i]
                    row[f"{col['name']} (Invoice 1)"] = pair['col1']
                    row[f"{col['name']} (Invoice 2)"] = pair['col2']
                table_data.append(row)
            
            return JsonResponse({
                'headers': headers,
                'table_data': table_data,
                'total_rows': max_len,
                'mapping_type': 'fuzzy_paired',
                'pair_separators': [i*2+1 for i in range(len(combined_columns))]  # Mark positions after each pair
            })
        
        else:
            # Handle multiple column pairs with proper row matching
            column_pairs = body.get("column_pairs", [])
            
            if not column_pairs:
                return JsonResponse({'error': 'No column pairs provided'}, status=400)
            
            billing1 = request.session.get("billing1", [])
            billing2 = request.session.get("billing2", [])
            
            if not billing1 or not billing2:
                return JsonResponse({'error': 'No billing data found'}, status=400)
            
            df1 = pd.DataFrame(billing1)
            df2 = pd.DataFrame(billing2)
            
            # Validate columns exist
            for pair in column_pairs:
                col1, col2 = pair['column1'], pair['column2']
                if col1 not in df1.columns:
                    return JsonResponse({'error': f'Column "{col1}" not found in invoice 1'}, status=400)
                if col2 not in df2.columns:
                    return JsonResponse({'error': f'Column "{col2}" not found in invoice 2'}, status=400)
            
            # Create comparison data maintaining row structure
            max_len = max(len(df1), len(df2))
            comparison_data = []
            
            for i in range(max_len):
                row_data = {'row_index': i + 1}
                
                for pair in column_pairs:
                    col1, col2 = pair['column1'], pair['column2']
                    
                    val1 = str(df1.iloc[i][col1]) if i < len(df1) else ''
                    val2 = str(df2.iloc[i][col2]) if i < len(df2) else ''
                    
                    # Clean values
                    val1 = val1.strip() if val1 != 'nan' else ''
                    val2 = val2.strip() if val2 != 'nan' else ''
                    
                    row_data[f'{col1}_invoice1'] = val1
                    row_data[f'{col2}_invoice2'] = val2
                    
                    # Determine match status
                    if val1 == '' and val2 == '':
                        match_status = ''
                    elif val1 == '' or val2 == '':
                        match_status = 'no'
                    else:
                        # Try numeric comparison first
                        try:
                            num1 = float(val1.replace(',', '').replace('$', ''))
                            num2 = float(val2.replace(',', '').replace('$', ''))
                            match_status = 'yes' if abs(num1 - num2) < 0.01 else 'no'
                        except ValueError:
                            # String comparison with fuzzy matching
                            score = fuzz.token_sort_ratio(val1.lower(), val2.lower())
                            match_status = 'yes' if score >= 85 else 'no'
                    
                    row_data[f'{col1}_{col2}_match'] = match_status
                
                comparison_data.append(row_data)
            
            return JsonResponse({
                'column_pairs': column_pairs,
                'comparison_data': comparison_data,
                'total_rows': max_len
            })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def build_mixed_table(request):
    """API to create a unified table by merging selected columns and mapping similar columns"""
    if request.method == "POST":
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON body'}, status=400)
        
        column_pairs = body.get("column_pairs", [])
        
        if not column_pairs:
            return JsonResponse({'error': 'No column pairs selected'}, status=400)
        
        billing1 = request.session.get("billing1", [])
        billing2 = request.session.get("billing2", [])
        
        if not billing1 or not billing2:
            return JsonResponse({'error': 'No billing data found'}, status=400)
        
        df1 = pd.DataFrame(billing1)
        df2 = pd.DataFrame(billing2)
        
        mixed_table_data = []
        headers = []
        max_rows = 0
        
        # First pass: determine headers and max rows
        for pair in column_pairs:
            col1 = pair.get('invoice1_column')
            col2 = pair.get('invoice2_column')
            unified_name = pair.get('unified_name', col1 or col2)
            
            if col1 and col1 in df1.columns:
                headers.append(f"{unified_name} (Invoice 1)")
                max_rows = max(max_rows, len(df1))
            if col2 and col2 in df2.columns:
                headers.append(f"{unified_name} (Invoice 2)")
                max_rows = max(max_rows, len(df2))
        
        # Initialize table data
        mixed_table_data = [[] for _ in range(max_rows)]
        
        # Second pass: populate data
        for pair in column_pairs:
            col1 = pair.get('invoice1_column')
            col2 = pair.get('invoice2_column')
            
            if col1 and col1 in df1.columns:
                col1_data = df1[col1].fillna('').astype(str).tolist()
                for i in range(max_rows):
                    value = col1_data[i] if i < len(col1_data) else ''
                    mixed_table_data[i].append(value)
            
            if col2 and col2 in df2.columns:
                col2_data = df2[col2].fillna('').astype(str).tolist()
                for i in range(max_rows):
                    value = col2_data[i] if i < len(col2_data) else ''
                    mixed_table_data[i].append(value)
        
        # Ensure all rows have the same number of columns
        max_cols = len(headers)
        for row in mixed_table_data:
            while len(row) < max_cols:
                row.append('')
        
        # Convert to list of dictionaries for JSON response
        table_rows = []
        for row_data in mixed_table_data:
            row_dict = {}
            for i, header in enumerate(headers):
                row_dict[header] = row_data[i] if i < len(row_data) else ''
            table_rows.append(row_dict)
        
        return JsonResponse({
            'headers': headers,
            'table_data': table_rows,
            'total_rows': len(table_rows)
        })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)



@csrf_exempt
def final_result_mapper(request):
    """API to map invoice rows using fuzzy matching and compare attributes exactly as shown in example"""
    if request.method == "POST":
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON body'}, status=400)
        
        key_column1 = body.get("key_column1", "").strip()
        key_column2 = body.get("key_column2", "").strip()
        
        if not key_column1 or not key_column2:
            return JsonResponse({'error': 'Both key columns must be selected'}, status=400)
        
        billing1 = request.session.get("billing1", [])
        billing2 = request.session.get("billing2", [])
        
        if not billing1 or not billing2:
            return JsonResponse({'error': 'No billing data found'}, status=400)
        
        df1 = pd.DataFrame(billing1)
        df2 = pd.DataFrame(billing2)
        
        if key_column1 not in df1.columns:
            return JsonResponse({'error': f'Column "{key_column1}" not found in invoice 1'}, status=400)
        
        if key_column2 not in df2.columns:
            return JsonResponse({'error': f'Column "{key_column2}" not found in invoice 2'}, status=400)
        
        # Get all rows from both invoices
        rows1 = df1.to_dict('records')
        rows2 = df2.to_dict('records')
        
        # Create mapping results
        paired_results = []
        used_rows2 = set()
        
        # Map each row from invoice 1 to best match in invoice 2
        for i, row1 in enumerate(rows1):
            key1 = str(row1.get(key_column1, '')).strip()
            best_match = None
            best_score = 0
            best_idx = -1
            
            # Skip empty keys
            if not key1:
                continue
                
            # Find best fuzzy match in invoice 2
            for j, row2 in enumerate(rows2):
                if j in used_rows2:
                    continue
                    
                key2 = str(row2.get(key_column2, '')).strip()
                if not key2:
                    continue
                    
                score = fuzz.token_sort_ratio(key1.lower(), key2.lower())
                
                if score > best_score and score >= 70:  # 70% similarity threshold
                    best_match = row2
                    best_score = score
                    best_idx = j
            
            # Create paired result
            if best_match:
                used_rows2.add(best_idx)
                
                # Compare attributes to determine match status
                mismatched_attrs = []
                
                # Get all columns except the key columns for comparison
                cols1 = [col for col in df1.columns if col != key_column1]
                cols2 = [col for col in df2.columns if col != key_column2]
                
                # Find matching column pairs using fuzzy matching
                column_pairs = []
                for col1 in cols1:
                    best_col2 = None
                    best_col_score = 0
                    for col2 in cols2:
                        col_score = fuzz.token_sort_ratio(col1.lower(), col2.lower())
                        if col_score > best_col_score and col_score >= 70:
                            best_col2 = col2
                            best_col_score = col_score
                    if best_col2:
                        column_pairs.append((col1, best_col2))
                
                # Compare values in matched column pairs
                for col1, col2 in column_pairs:
                    val1 = str(row1.get(col1, '')).strip()
                    val2 = str(best_match.get(col2, '')).strip()
                    
                    if val1 and val2:
                        # Try numeric comparison first
                        try:
                            num1 = float(val1.replace(',', '').replace('$', ''))
                            num2 = float(val2.replace(',', '').replace('$', ''))
                            if abs(num1 - num2) > 0.01:
                                mismatched_attrs.append(f"{col1}/{col2}")
                        except ValueError:
                            # String comparison
                            if fuzz.token_sort_ratio(val1.lower(), val2.lower()) < 90:
                                mismatched_attrs.append(f"{col1}/{col2}")
                    elif val1 != val2:
                        mismatched_attrs.append(f"{col1}/{col2}")
                
                # Determine match status
                if mismatched_attrs:
                    match_status = f"unmatched({', '.join(mismatched_attrs)})"
                else:
                    match_status = "matched"
                
                paired_results.append({
                    'row1': row1,
                    'row2': best_match,
                    'key1': key1,
                    'key2': str(best_match.get(key_column2, '')).strip(),
                    'match_status': match_status,
                    'column_pairs': column_pairs
                })
            else:
                # No match found
                paired_results.append({
                    'row1': row1,
                    'row2': None,
                    'key1': key1,
                    'key2': '',
                    'match_status': 'no matched column exist',
                    'column_pairs': []
                })
        
        # Add unmatched rows from invoice 2
        for j, row2 in enumerate(rows2):
            if j not in used_rows2:
                key2 = str(row2.get(key_column2, '')).strip()
                if key2:  # Only add non-empty keys
                    paired_results.append({
                        'row1': None,
                        'row2': row2,
                        'key1': '',
                        'key2': key2,
                        'match_status': 'no matched column exist',
                        'column_pairs': []
                    })
        
        # Get all unique columns for display headers
        all_columns1 = [col for col in df1.columns if col != key_column1]
        all_columns2 = [col for col in df2.columns if col != key_column2]
        
        return JsonResponse({
            'key_column1': key_column1,
            'key_column2': key_column2,
            'columns1': all_columns1,
            'columns2': all_columns2,
            'paired_results': paired_results,
            'total_pairs': len(paired_results)
        })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def create_unified_table(request):
    """API to create a unified table by merging selected columns and mapping similar columns"""
    if request.method == "POST":
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON body'}, status=400)
        
        key_column1 = body.get("key_column1", "").strip()
        key_column2 = body.get("key_column2", "").strip()
        
        if not key_column1 or not key_column2:
            return JsonResponse({'error': 'Both key columns must be selected'}, status=400)
        
        billing1 = request.session.get("billing1", [])
        billing2 = request.session.get("billing2", [])
        
        if not billing1 or not billing2:
            return JsonResponse({'error': 'No billing data found'}, status=400)
        
        df1 = pd.DataFrame(billing1)
        df2 = pd.DataFrame(billing2)
        
        if key_column1 not in df1.columns:
            return JsonResponse({'error': f'Column "{key_column1}" not found in invoice 1'}, status=400)
        
        if key_column2 not in df2.columns:
            return JsonResponse({'error': f'Column "{key_column2}" not found in invoice 2'}, status=400)
        
        # Function to find similar columns using fuzzy matching
        def find_similar_columns(df1_cols, df2_cols, exclude_cols):
            column_mapping = {}
            used_df2_cols = set()
            
            for col1 in df1_cols:
                if col1 in exclude_cols:
                    continue
                    
                best_match = None
                best_score = 0
                
                for col2 in df2_cols:
                    if col2 in exclude_cols or col2 in used_df2_cols:
                        continue
                    
                    # Check similarity using fuzzy matching
                    score = fuzz.token_sort_ratio(col1.lower(), col2.lower())
                    
                    # Also check against COLUMN_MAPPING
                    for key, synonyms in COLUMN_MAPPING.items():
                        if any(fuzz.token_sort_ratio(col1.lower(), syn.lower()) >= 70 for syn in synonyms):
                            if any(fuzz.token_sort_ratio(col2.lower(), syn.lower()) >= 70 for syn in synonyms):
                                score = max(score, 85)  # Boost score for synonym matches
                    
                    if score > best_score and score >= 70:
                        best_match = col2
                        best_score = score
                
                if best_match:
                    column_mapping[col1] = best_match
                    used_df2_cols.add(best_match)
            
            return column_mapping
        
        # Find column mappings (excluding the key columns)
        exclude_cols = {key_column1, key_column2}
        column_mapping = find_similar_columns(df1.columns, df2.columns, exclude_cols)
        
        # Get all unique columns for the unified table
        unified_columns = [key_column1]  # Start with the key column
        
        # Add mapped columns
        for col1, col2 in column_mapping.items():
            unified_columns.append(col1)
        
        # Add unmapped columns from df1
        for col in df1.columns:
            if col not in exclude_cols and col not in column_mapping and col not in unified_columns:
                unified_columns.append(col)
        
        # Add unmapped columns from df2
        for col in df2.columns:
            if col not in exclude_cols and col not in column_mapping.values() and col not in unified_columns:
                unified_columns.append(col)
        
        # Create unified table data
        unified_data = []
        
        # Add rows from Invoice 1
        for _, row in df1.iterrows():
            unified_row = {}
            unified_row[key_column1] = str(row[key_column1]) if pd.notna(row[key_column1]) else ''
            
            for col in unified_columns[1:]:  # Skip key column as it's already added
                if col in df1.columns:
                    unified_row[col] = str(row[col]) if pd.notna(row[col]) else ''
                else:
                    unified_row[col] = ''
            
            unified_data.append(unified_row)
        
        # Add rows from Invoice 2
        for _, row in df2.iterrows():
            unified_row = {}
            unified_row[key_column1] = str(row[key_column2]) if pd.notna(row[key_column2]) else ''
            
            for col in unified_columns[1:]:  # Skip key column as it's already added
                if col in column_mapping and column_mapping[col] in df2.columns:
                    # Use mapped column from df2
                    mapped_col = column_mapping[col]
                    unified_row[col] = str(row[mapped_col]) if pd.notna(row[mapped_col]) else ''
                elif col in df2.columns:
                    # Direct column match
                    unified_row[col] = str(row[col]) if pd.notna(row[col]) else ''
                else:
                    unified_row[col] = ''
            
            unified_data.append(unified_row)
        
        return JsonResponse({
            'unified_columns': unified_columns,
            'unified_data': unified_data,
            'column_mapping': column_mapping,
            'total_rows': len(unified_data),
            'invoice1_rows': len(df1),
            'invoice2_rows': len(df2)
        })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)
