from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_csv, name='upload_csv'),
    path('clear/', views.clear_data, name='clear_data'),
    path('get-gpt-analysis/', views.get_gpt_analysis, name='get_gpt_analysis'),
    path('execute-sql/', views.execute_sql_query, name='execute_sql_query'),
    path('generate-sql/', views.generate_sql_query, name='generate_sql_query'),
    path('fix-sql/', views.fix_sql_query, name='fix_sql_query'),
    path('submit-query-feedback/', views.submit_query_feedback, name='submit_query_feedback'),
] 