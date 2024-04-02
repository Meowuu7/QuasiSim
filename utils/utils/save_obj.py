def save_obj(path, v, f, c=None):
    with open(path, 'w') as file:
        if c is None:
            for i in range(v.shape[0]):
                file.write('v {} {} {}\n'.format(v[i, 0], v[i, 1], v[i, 2]))
        else:
            for i in range(v.shape[0]):
                file.write('v {} {} {} {} {} {}\n'.format(v[i, 0], v[i, 1], v[i, 2], c[i, 0], c[i, 1], c[i, 2]))
        for i in range(f.shape[0]):
            file.write('f {} {} {}\n'.format(f[i, 0] + 1, f[i, 1] + 1, f[i, 2] + 1))