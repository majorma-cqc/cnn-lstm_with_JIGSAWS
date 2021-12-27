import pdb

print("Hello.\n")
pdb.set_trace()
for i in range(10):
    pdb.set_trace()
    print(i ** (i + 1))
    i = i + 1
    
pdb.set_trace()
print("good bye")
pass