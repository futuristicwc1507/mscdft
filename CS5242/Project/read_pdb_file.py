def read_pdb(filename):
    
    with open(filename, 'r') as file:
        strline_L = file.readlines()

    strline_L=[strline.strip() for strline in strline_L]

    X_list=[float(strline.split()[-3]) for strline in strline_L]
    Y_list=[float(strline.split()[-2]) for strline in strline_L]
    Z_list=[float(strline.split()[-1]) for strline in strline_L]
    atomtype_list=[strline.split()[2] for strline in strline_L]
    

    return X_list, Y_list, Z_list, atomtype_list


X_list, Y_list, Z_list, atomtype_list=read_pdb("project_test_data/pdbs/1A0Q.pdb")
print(X_list)
print(Y_list)
print(Z_list)
print(atomtype_list)
