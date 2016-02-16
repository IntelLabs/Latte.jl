import os
import subprocess
import sys

def untar(file, output_dir):
    subprocess.check_call("tar -xf {file} -C {output_dir}".format(file=file, output_dir=output_dir).split())


for dataset in ["ILSVRC2012_img_train.tar", "ILSVRC2012_img_test.tar"]:
    dataset_name, _ = dataset.split(".")
    os.mkdir("{dset}".format(dset=dataset_name))

    print("Unpacking toplevel tar")
    untar(dataset, dataset_name)

    classes = os.listdir("./{dir}".format(dir=dataset_name))

    for cls in classes:
        print("Unpacking file {name}".format(name=cls))
        label, _ = cls.split(".")
        output_dir = "./{dir}/{label}".format(dir=dataset_name, label=label)
        os.mkdir(output_dir)
        tar_file = "./{dir}/{file}".format(dir=dataset_name, file=cls)
        untar(tar_file, output_dir)
        os.remove(tar_file)
