def replace(name):
        pos = name.rfind("\\")
        if pos > -1:
              name = name[:pos]  + name[pos +1: ]
        return name

def delete(string,substring):
       str_list = string.split(substring)
       output_string = ""
       for element in str_list:
             output_string += element
       return output_string
