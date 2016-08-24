# -*- coding: utf-8 -*-

from bottle import route, run, template, get, post, request, static_file
import similarity
import train_svm
import bsetresponse as best
from itertools import chain




home = open('boot/home.html','r').read()
out = open('boot/output.html', 'r').read()
res = open('boot/response.html', 'r').read()






### main func ###



@get('/')
def index():
    return home
#    return r_top, form.format(''), r_bottom

@post('/') # or @route('/login', method='POST')
def do_index():
    new_input = request.forms.get('text')
    final_res = similarity.adjusted_sim(str(new_input))
    str_res = []
    for i,j in final_res:
        str_i = i.encode('utf8')
        str_j = str(j)
        str_res.append([str_i,str_j])

    tab = ['\t'.join(i) for i in str_res]
    final = '<br>'.join(tab)
    return out.format(str(new_input), final, str(new_input))



@get('/response')
def response():
    redirect('/')



@post('/response')
def response():
    new_input = request.forms.get('text')
    label = train_svm.sent_score(str(new_input))
    result = best.res(label)
    random_res = '<br>'.join(result)
    return res.format(random_res)


### end main func ###


### apply html templates and relative files ###

@route('/<filepath:path>')
def server_static(filepath):
	return static_file(filepath, root='boot/')


run(host='localhost', port=8000, debug=True)
