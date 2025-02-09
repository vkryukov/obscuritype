from django.shortcuts import render, redirect
import pandas as pd
import io

# Create your views here.

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
            
            # Get column names
            columns = csv_data.columns.tolist()
            
            # Store in session
            request.session['csv_columns'] = columns
            request.session['csv_filename'] = csv_file.name
            request.session['csv_data'] = file_data
            
            # Redirect to GET request
            return redirect('/')
            
        except Exception as e:
            context['error_message'] = f'Error parsing CSV file: {str(e)}'
    else:
        # Check if we have data in session
        if 'csv_columns' in request.session:
            context['columns'] = request.session['csv_columns']
            context['filename'] = request.session['csv_filename']
    
    return render(request, 'uploader/upload.html', context)
