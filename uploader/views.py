from django.shortcuts import render
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
            context['columns'] = columns
            
        except Exception as e:
            context['error_message'] = f'Error parsing CSV file: {str(e)}'
    
    return render(request, 'uploader/upload.html', context)
