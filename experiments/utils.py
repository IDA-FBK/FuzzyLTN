def save_list_in_a_file(list_to_save, path_to_file):
    with open(path_to_file, "w") as f:
        for el_list in list_to_save:
            f.write(el_list + "\n")
            print(el_list)
