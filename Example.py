a=10
b=4
print("a+b:",a+b)
print("a-b:",a-b)
print("a*b:",a*b)
print("a/b:",a/b)
print("a//b:",a//b)
print("a**b:",a**b)
c=int(input("Enter a value:"))
print("Multiply value with 2:",c*2)
if False:
    print("in False")
else:
    print("in Else")

if a>b:
    print("a is bigger")
elif b>a:
    print("b is bigger")
else:
    print("a equals b")

print("Enter value:")
a=int(input("First value:"))
b=int(input("Second value:"))
c=int(input("Third value:"))
biggest=a
if b>biggest:
    biggest=b
elif c>biggest:
    biggest=c
print("Biggest value is:",biggest)

for i in range(5,10):
    print(i,i*i)
x=0
while x<10:
    print("while",x)
    x=x+1
