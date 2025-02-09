from django.db import models

# Create your models here.

class QueryFeedback(models.Model):
    original_request = models.TextField()
    generated_query = models.TextField()
    query_result = models.TextField()
    raw_data_excerpt = models.TextField()
    user_prompt = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Feedback for query: {self.generated_query[:50]}..."
