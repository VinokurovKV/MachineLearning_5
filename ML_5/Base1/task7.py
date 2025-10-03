def find_modified_max_argmax(L,f):
 a=[f(x)for x in L if type(x)==int]
 return a and(max(a),a.index(max(a)))or()