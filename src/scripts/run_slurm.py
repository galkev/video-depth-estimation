import argparse
import sys
import subprocess
import os

if __name__ == "__main__":
    slurm_support_scripts = [
        "train_model.py",
        "train_model_json.py",
        "test.py",
        # "blender_script.py"
    ]

    slurm_log_dir = "/storage/slurm/galim/Documents/master-thesis/logs/slurm_logs"

    parser = argparse.ArgumentParser()

    argv = sys.argv
    if "--" not in argv:
        argv_slurm = []  # as if no args are passed
        argv_cmd = argv[1:]
    else:
        argv_slurm = argv[:argv.index("--")]
        argv_cmd = argv[argv.index("--") + 1:]

    prog_name = argv_cmd[0]

    parser.add_argument("--sbatch", default="sbatch/default.sbatch")
    parser.add_argument("--job_name", default=" ".join(argv_cmd))

    args, _ = parser.parse_known_args(argv_slurm)

    if prog_name.endswith(".py"):
        interpreter = "python"
    else:
        interpreter = ""

    if prog_name in slurm_support_scripts:
        add_args = ["--job_id", "${SLURM_JOBID}", "--job_name", args.job_name]
    else:
        add_args = []  # if slurm support add job id

    sbatch_cmd = [
        "sbatch",
        "--parsable",
        "-o", slurm_log_dir + "/slurm-%j.out",
        "-J", args.job_name,
        args.sbatch,
        interpreter, *argv_cmd, *add_args
    ]

    job_id = subprocess.check_output(sbatch_cmd).decode("ascii").strip()

    slurm_out_file = "{}/slurm-{}.out".format(
        slurm_log_dir,
        job_id
    )

    cmd = "echo 'Job {} started. Listening'; touch {}; tail -f {}".format(
        job_id,
        slurm_out_file,
        slurm_out_file
    )

    os.system(cmd)
