from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
import io

# Create your views here.

def detect_column_types(df, sample_rows=50):
    # Use only first n rows for type detection
    df_sample = df.head(sample_rows)
    column_types = {}
    
    for column in df.columns:
        series = df_sample[column]
        unique_count = series.nunique()
        total_count = len(series.dropna())
        
        # Check if boolean
        if set(series.dropna().unique()) <= {0, 1, True, False, 'true', 'false', 'True', 'False', 'TRUE', 'FALSE'}:
            column_types[column] = 'boolean'
        # Check if categorical (less than 10% unique values)
        elif unique_count < total_count * 0.1 or unique_count < 10:
            column_types[column] = 'categorical'
        # Check if integer
        elif series.dtype in ['int64', 'int32']:
            column_types[column] = 'integer'
        # Check if float
        elif series.dtype in ['float64', 'float32']:
            column_types[column] = 'float'
        # Check if datetime
        elif series.dtype in ['datetime64[ns]'] or pd.to_datetime(series, errors='coerce').notna().all():
            column_types[column] = 'datetime'
        # If most values are long strings (>100 chars), consider it text
        elif series.dtype == 'object' and series.str.len().mean() > 100:
            column_types[column] = 'text'
        else:
            column_types[column] = 'string'
    
    return column_types

def group_columns_by_type(columns_dict):
    groups = {
        'boolean': [],
        'categorical': [],
        'integer': [],
        'float': [],
        'datetime': [],
        'text': [],
        'string': []
    }
    
    for column, type_ in columns_dict.items():
        groups[type_].append(column)
    
    # Sort columns within each group
    for group in groups.values():
        group.sort()
    
    # Remove empty groups
    return {k: v for k, v in groups.items() if v}

def upload_csv(request):
    context = {}
    
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        
        if not csv_file.name.endswith('.csv'):
            context['error_message'] = 'Please upload a CSV file.'
            return render(request, 'uploader/upload.html', context)
        
        try:
            # Read the uploaded file into memory
            file_data = csv_file.read().decode('utf-8')
            csv_data = pd.read_csv(io.StringIO(file_data))
            
            # Analyze column types
            column_types = detect_column_types(csv_data)
            grouped_columns = group_columns_by_type(column_types)
            
            # Store in session
            request.session['grouped_columns'] = grouped_columns
            request.session['csv_filename'] = csv_file.name
            request.session['csv_data'] = file_data
            
            # Redirect to GET request
            return redirect('/')
            
        except Exception as e:
            context['error_message'] = f'Error parsing CSV file: {str(e)}'
    else:
        # Check if we have data in session
        if 'grouped_columns' in request.session:
            context['grouped_columns'] = request.session['grouped_columns']
            context['filename'] = request.session['csv_filename']
    
    return render(request, 'uploader/upload.html', context)
