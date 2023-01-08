from django.http import HttpResponse
from django.shortcuts import render
from django.template.response import TemplateResponse
from django.core.exceptions import ObjectDoesNotExist
import traceback
import logging

class SimpleMiddleware:

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        return response

    def process_exception(self, request, exception):
        print(exception)
        context = {
            'exception': exception
        }
        if isinstance(exception, ValueError):
            context = dict(confict_details=str(exception))
            return TemplateResponse(request, 'img_trans/409.html', context, status=409)
        elif isinstance(exception, ValueError):
            pass  
        elif isinstance(exception, IndexError):
            pass   
        elif isinstance(exception, ObjectDoesNotExist):
            return render(request, 'img_trans/error.html', context)          
        else:
            return render(request, 'img_trans/error.html', context)



class SimpleMiddleware:

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        return response

    def process_exception(self, request, exception):
        err = traceback.format_exc()
        logging.error(err)
        logging.error(exception)
