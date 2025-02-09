from django.shortcuts import render, redirect
from django.http import JsonResponse
import pandas as pd
import numpy as np
import io
import json
import ast
import os
from openai import OpenAI
from pandasql import sqldf

# Create your views here.

def is_list_column(series):
    """Check if a column contains list-like strings."""
    try:
        # Check first non-null value
        first_val = series.dropna().iloc[0]
        return isinstance(first_val, str) and first_val.startswith('[') and first_val.endswith(']')
    except:
        return False

def is_numeric_column(series):
    """Check if a column contains numeric values, even if stored as strings."""
    try:
        # Try converting to numeric, if it works, it's numeric
        pd.to_numeric(series, errors='raise')
        return True
    except:
        return False

def is_boolean_series(series):
    """Check if a series contains only boolean values."""
    try:
        unique_vals = set()
        for val in series.dropna():
            if isinstance(val, (str, int, bool)):
                if isinstance(val, str):
                    val = val.lower()
                unique_vals.add(str(val))
        return unique_vals <= {'0', '1', 'true', 'false'}
    except:
        return False

def detect_column_types(df, sample_rows=1000):
    """Enhanced column type detection with larger sample size and improved numerical detection."""
    df_sample = df.head(sample_rows)
    column_types = {}
    
    for column in df.columns:
        series = df_sample[column]
        
        # Skip empty columns
        if len(series.dropna()) == 0:
            column_types[column] = 'empty'
            continue
            
        # First, check if it's already a list or list-like string
        if isinstance(series.iloc[0], list) or is_list_column(series):
            column_types[column] = 'list'
            continue
        
        # For non-list columns, we can safely calculate unique counts
        unique_count = series.nunique()
        total_count = len(series.dropna())
        
        # Check if numeric (including string-stored numbers)
        if is_numeric_column(series):
            # Get all unique values as floats for analysis
            unique_vals = pd.to_numeric(series.dropna().unique())
            
            # Check if all values are 0, 1, or boolean-like
            is_potential_boolean = all(val in [0, 1, 0.0, 1.0] for val in unique_vals)
            
            if not is_potential_boolean:
                # If we have non-boolean values, determine if integer or float
                if all(float(x).is_integer() for x in series.dropna()):
                    column_types[column] = 'integer'
                else:
                    column_types[column] = 'float'
                continue
        
        # Check if boolean (only if we haven't determined it's a non-boolean numeric)
        if is_boolean_series(series):
            column_types[column] = 'boolean'
            continue
            
        # Check if datetime
        try:
            if pd.to_datetime(series, errors='raise').notna().all():
                column_types[column] = 'datetime'
                continue
        except:
            pass
            
        # Check if categorical
        if (unique_count < 10) or (total_count > 100 and (unique_count / total_count) < 0.05):
            column_types[column] = 'categorical'
            continue
            
        # If most values are long strings (>100 chars), consider it text
        if series.dtype == 'object' and series.str.len().mean() > 100:
            column_types[column] = 'text'
            continue
            
        # Default to string
        column_types[column] = 'string'
    
    return column_types

def group_columns_by_type(columns_dict):
    groups = {
        'list': [],
        'boolean': [],
        'categorical': [],
        'integer': [],
        'float': [],
        'datetime': [],
        'text': [],
        'string': [],
        'empty': []
    }
    
    for column, type_ in columns_dict.items():
        groups[type_].append(column)
    
    # Sort columns within each group
    for group in groups.values():
        group.sort()
    
    # Remove empty groups
    return {k: v for k, v in groups.items() if v}

def analyze_data_with_gpt(data_sample):
    """Analyze a sample of the data using GPT-4 to get insights."""
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Convert the data sample to a string format
        data_str = json.dumps(data_sample, indent=2)
        
        prompt = f"""Analyze this dataset (showing first few rows) and provide a concise 2-3 paragraph summary.
Focus on:
1. What kind of data this appears to be
2. Key columns and their meaning
3. Any interesting patterns or characteristics

Here's the data:
{data_str}

Be concise but insightful. Limit response to 2-3 paragraphs."""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analyst providing concise insights about datasets."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing data: {str(e)}"

def get_gpt_analysis(request):
    """AJAX endpoint to get GPT analysis of the data."""
    if 'data_json' not in request.session:
        return JsonResponse({'error': 'No data available'}, status=400)
    
    try:
        # First check if we already have the analysis
        if 'data_analysis' in request.session:
            return JsonResponse({'analysis': request.session['data_analysis']})
        
        # If not, generate new analysis
        data = json.loads(request.session['data_json'])
        data_sample = data[:10]  # Get first 10 rows
        analysis = analyze_data_with_gpt(data_sample)
        
        # Store the analysis in session
        request.session['data_analysis'] = analysis
        
        return JsonResponse({'analysis': analysis})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def execute_sql_query(request):
    """Execute SQL query on the uploaded data."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    if 'data_json' not in request.session:
        return JsonResponse({'error': 'No data available. Please upload a CSV file first.'}, status=400)
    
    try:
        query = request.POST.get('query')
        if not query:
            return JsonResponse({'error': 'No query provided'}, status=400)
        
        # Load data from session
        data = pd.DataFrame(json.loads(request.session['data_json']))
        
        # Execute the query
        result = sqldf(query, locals())
        
        # Convert result to JSON
        result_json = json.loads(result.to_json(orient='records', date_format='iso'))
        
        # Get column names for the table headers
        columns = list(result.columns)
        
        return JsonResponse({
            'success': True,
            'data': result_json,
            'columns': columns
        })
    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=400)

def upload_csv(request):
    print("\n=== Starting upload_csv view ===")
    print(f"Request method: {request.method}")
    context = {}
    
    # Check if we have all required data in session first
    required_keys = ['grouped_columns', 'csv_filename', 'data_json']
    print(f"Session keys present: {list(request.session.keys())}")
    
    if all(key in request.session for key in required_keys):
        print("All required session data found")
        context['grouped_columns'] = request.session['grouped_columns']
        context['filename'] = request.session['csv_filename']
        context['data'] = json.loads(request.session['data_json'])
        if 'data_analysis' in request.session:
            # If we have cached analysis, use it directly
            context['data_analysis'] = request.session['data_analysis']
        else:
            # Otherwise show loading state
            context['show_analysis_loading'] = True
        print(f"Loaded from session - filename: {context['filename']}")
    else:
        print("Missing some session data")
        for key in required_keys:
            print(f"  {key}: {'present' if key in request.session else 'missing'}")
    
    # Handle file upload
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        print(f"\nProcessing uploaded file: {csv_file.name}")
        
        if not csv_file.name.endswith('.csv'):
            print("Error: Not a CSV file")
            context['error_message'] = 'Please upload a CSV file.'
            return render(request, 'uploader/upload.html', context)
        
        try:
            print("Reading file contents...")
            file_data = csv_file.read().decode('utf-8')
            
            print("Parsing CSV with pandas...")
            csv_data = pd.read_csv(io.StringIO(file_data), keep_default_na=False)
            print(f"CSV loaded successfully. Shape: {csv_data.shape}")
            
            print("Processing list columns...")
            for col in csv_data.columns:
                if is_list_column(csv_data[col]):
                    print(f"Converting list column: {col}")
                    csv_data[col] = csv_data[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
            
            print("Detecting column types...")
            column_types = detect_column_types(csv_data)
            print("Column types detected:", column_types)
            
            grouped_columns = group_columns_by_type(column_types)
            print("Columns grouped by type:", grouped_columns)
            
            print("Preparing JSON data...")
            json_data = csv_data.copy()
            for col in json_data.columns:
                if column_types[col] == 'list':
                    print(f"Converting list column for JSON: {col}")
                    json_data[col] = json_data[col].apply(str)
            
            data_json = json_data.head(1000).to_json(orient='records', date_format='iso')
            print("JSON data prepared successfully")
            
            print("Storing data in session...")
            request.session['grouped_columns'] = grouped_columns
            request.session['csv_filename'] = csv_file.name
            request.session['data_json'] = data_json
            # Clear any existing analysis when uploading new file
            if 'data_analysis' in request.session:
                del request.session['data_analysis']
            print("Session data stored successfully")
            
            print("Updating context...")
            context['grouped_columns'] = grouped_columns
            context['filename'] = csv_file.name
            context['data'] = json.loads(request.session['data_json'])
            context['show_analysis_loading'] = True
            print("Context updated successfully")
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            import traceback
            print("Traceback:", traceback.format_exc())
            context['error_message'] = f'Error processing file: {str(e)}'
    
    print("\nFinal context keys:", list(context.keys()))
    print("=== Ending upload_csv view ===\n")
    return render(request, 'uploader/upload.html', context)

def clear_data(request):
    """Clear all session data and redirect to the upload page."""
    if request.method == 'POST':
        # Clear all relevant session data
        keys_to_clear = ['grouped_columns', 'csv_filename', 'data_json', 'csv_columns', 'csv_data']
        for key in keys_to_clear:
            if key in request.session:
                del request.session[key]
    return redirect('upload_csv')

def generate_sql_query(request):
    """Generate SQL query from natural language using GPT."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    if 'data_json' not in request.session:
        return JsonResponse({'error': 'No data available'}, status=400)
    
    try:
        query = request.POST.get('query')
        if not query:
            return JsonResponse({'error': 'No query provided'}, status=400)
        
        # Get column information and data analysis
        grouped_columns = request.session.get('grouped_columns', {})
        data_analysis = request.session.get('data_analysis', '')
        
        # Create a description of available columns
        columns_desc = []
        for type_name, columns in grouped_columns.items():
            if columns:
                columns_desc.append(f"{type_name.title()} columns: {', '.join(columns)}")
        
        prompt = f"""Given a dataset with the following structure:

{chr(10).join(columns_desc)}

Dataset description:
{data_analysis}

Convert this natural language query into a SQL query:
"{query}"

Return ONLY the SQL query, nothing else. The table name is 'data'.
Make sure to handle NULL values appropriately and use proper SQL syntax."""

        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a SQL expert that converts natural language queries into SQL. Return only the SQL query, nothing else."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        sql_query = response.choices[0].message.content.strip()
        
        return JsonResponse({
            'success': True,
            'sql_query': sql_query
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

def fix_sql_query(request):
    """Fix a SQL query that produced an error."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    try:
        original_query = request.POST.get('original_query')
        sql_query = request.POST.get('sql_query')
        error_message = request.POST.get('error_message')
        
        if not all([original_query, sql_query, error_message]):
            return JsonResponse({'error': 'Missing required parameters'}, status=400)
        
        prompt = f"""The following SQL query was generated for this natural language request:
"{original_query}"

The generated SQL query was:
{sql_query}

But it produced this error:
{error_message}

Please fix the SQL query to resolve this error. Return ONLY the fixed SQL query, nothing else.
The table name should be 'data'."""

        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a SQL expert that fixes SQL queries. Return only the fixed SQL query, nothing else."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        fixed_query = response.choices[0].message.content.strip()
        
        return JsonResponse({
            'success': True,
            'sql_query': fixed_query
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
