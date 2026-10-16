import sys
def bf(code):
    tape=[0]*300; p=0; i=0; out=''; st=[]; jump={}
    for j,c in enumerate(code):
        if c=='[': st.append(j)
        elif c==']': k=st.pop(); jump[k]=j; jump[j]=k
    while i<len(code):
        c=code[i]
        if c=='>': p+=1
        elif c=='<': p-=1
        elif c=='+': tape[p]=(tape[p]+1)%256
        elif c=='-': tape[p]=(tape[p]-1)%256
        elif c=='.': out+=chr(tape[p])
        elif c=='[' and tape[p]==0: i=jump[i]
        elif c==']' and tape[p]!=0: i=jump[i]
        i+=1
    return out
print(repr(bf(sys.argv[1])))
