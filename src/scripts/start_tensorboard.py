import os
import argparse


def main():
    ssh_host = "atcremers17"
    virt_env = "torch_tfn"
    target_port = 7000

    parser = argparse.ArgumentParser()

    parser.add_argument("-t", action="store_true")
    parser.add_argument("-l", action="store_true")

    args = parser.parse_args()

    if not args.l:
        # tb_logdir = "/storage/slurm/galim/Documents/master-thesis/logs/tblogs"
        tb_logdir = "~/Documents/master-thesis/logs/tblogs"

        print("http://localhost:6006")

        cmd = "nohup ssh -N -f -L localhost:6006:localhost:{} {};".format(target_port, ssh_host)

        if not args.t:
            cmd += \
                "ssh {} '" \
                ". miniconda3/etc/profile.d/conda.sh;" \
                "conda activate {};" \
                "killall -u galim tensorboard;" \
                "export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir={} --port={};"\
                "'".format(
                    ssh_host,
                    virt_env,
                    tb_logdir,
                    target_port
                )
    else:
        tb_logdir = "/home/kevin/Documents/master-thesis/logs/tblogs"

        cmd = \
            "killall tensorboard && " \
            "tensorboard --logdir={};".format(
                tb_logdir
            )

    os.system(cmd)


if __name__ == "__main__":
    main()
