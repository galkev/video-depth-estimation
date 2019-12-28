import os
import sys

if __name__ == "__main__":
    script_dir = "Documents/master-thesis/code/mt-project/scripts"
    ssh_host = "atcremers17"
    virt_env = "torch_tfn"
    sync_dir = "slurm_sync"

    slurm_job_args = " ".join(sys.argv[1:])

    cmd = ""

    cmd += "rsync -av {} {}:{};".format(
        sync_dir,
        ssh_host,
        script_dir
    )

    cmd += "ssh {} '"\
           ". miniconda3/etc/profile.d/conda.sh;" \
           "conda activate {};" \
           "cd {};" \
           "git pull;" \
           "python run_slurm.py {}" \
           "'".format(
            ssh_host,
            virt_env,
            script_dir,
            slurm_job_args
            )

    # print(cmd)
    os.system(cmd)
