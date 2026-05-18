import os
from pathlib import Path
import math
import shlex


# Site-specific values are read from the environment (see ``.env.example``).
# Partition is required; the others have generic defaults.
DEFAULT_SLURM_PARTITION = os.environ.get("MIRO_SLURM_PARTITION")
DEFAULT_SLURM_OUTPUT_DIR = os.environ.get(
    "MIRO_SLURM_OUTPUT_DIR",
    str(Path.cwd() / "slurm-logs"),
)
DEFAULT_PYTHONPATH = os.environ.get("MIRO_PYTHONPATH", "")


def _format_hydra_value(value):
    """Format a Python value for Hydra CLI overrides, quoting lists."""
    if isinstance(value, list):
        inner = ",".join(str(v) for v in value)
        return f"'[{inner}]'"
    return value


class SlurmExperiment:
    def __init__(
        self,
        exp_name,
        job_name,
        slurm_array_nb_jobs=None,
        max_simultaneous_jobs=None,
        sub_job_index=0,
        num_nodes=1,
        num_gpus_per_node=8,
        partition=None,
        cmd_path="train.py",
        time=None,
        launch_from_compute_node=False,
        min_time=None,
        root_dir=None,
        use_torchrun=False,
        stagger_delay=100,
    ):
        self.expname = exp_name
        self.job_name = job_name
        self.nodes = num_nodes
        self.num_gpus_per_node = num_gpus_per_node
        self.partition = partition if partition is not None else DEFAULT_SLURM_PARTITION
        if not self.partition:
            raise ValueError(
                "SLURM partition is required: either pass partition=... to "
                "SlurmExperiment(...) or export MIRO_SLURM_PARTITION."
            )
        self.slurm_array_nb_jobs = slurm_array_nb_jobs
        self.max_simultaneous_jobs = max_simultaneous_jobs
        self.sub_job_index = sub_job_index
        self.cmd_path = cmd_path
        self.time = time
        self.min_time = min_time
        self.launch_from_compute_node = launch_from_compute_node
        self.root_dir = root_dir
        self.use_torchrun = use_torchrun
        self.stagger_delay = stagger_delay

        # Initialize paths
        self.slurm_script_path = None
        self.slurm_out_path = None
        self.slurm_err_path = None
        self.slurm_job_id = None  # To store job ID after submission
        self.cmd = None  # For single command
        self.cmds = None  # For list of commands
        self.sequential_commands = None  # For sequential commands with different venvs
        self.sequential_commands_list = None  # For job array of sequential commands

    def build_cmd(
        self,
        hydra_args=None,
        hydra_arg_list=None,
        sequential_commands=None,
        sequential_commands_list=None,
    ):
        """Build commands for execution.

        Args:
            hydra_args: Dict of hydra arguments for single command or numeric array
            hydra_arg_list: List of dicts for command list array
            sequential_commands: List of dicts with 'command' and optional 'hydra_args'
                               for sequential execution
            sequential_commands_list: List of sequential_commands for job array of sequential executions
        """
        # Validate input: exactly one of the four options must be provided
        provided_args = [
            hydra_args is not None,
            hydra_arg_list is not None,
            sequential_commands is not None,
            sequential_commands_list is not None,
        ]
        if sum(provided_args) != 1:
            raise ValueError(
                "Provide exactly one of: hydra_args (single command/numeric array), "
                "hydra_arg_list (command list array), sequential_commands (sequential execution), "
                "or sequential_commands_list (job array of sequential executions)."
            )

        if hydra_arg_list is not None:
            if self.slurm_array_nb_jobs is not None:
                raise ValueError(
                    "Cannot provide slurm_array_nb_jobs during initialization when using hydra_arg_list in build_cmd."
                )
            self.cmds = []
            for args_dict in hydra_arg_list:
                hydra_modifiers = []
                for hydra_arg, value in args_dict.items():
                    if hydra_arg.startswith("--"):
                        hydra_modifiers.append(f" {hydra_arg} {value}")
                    else:
                        hydra_modifiers.append(f" {hydra_arg}={_format_hydra_value(value)}")
                command = f"{self.cmd_path} {''.join(hydra_modifiers)}"
                self.cmds.append(command)
                print(f"Prepared command: {command}")
            # Set slurm_array_nb_jobs based on the list length
            self.slurm_array_nb_jobs = len(self.cmds)
            if not self.slurm_array_nb_jobs:
                raise ValueError("hydra_arg_list cannot be empty.")
            print(f"Built {self.slurm_array_nb_jobs} commands for job array.")

        elif sequential_commands is not None:
            # Logic for sequential commands
            if self.slurm_array_nb_jobs is not None:
                raise ValueError(
                    "Cannot provide slurm_array_nb_jobs during initialization when using sequential_commands."
                )
            if not isinstance(sequential_commands, list) or not sequential_commands:
                raise ValueError("sequential_commands must be a non-empty list.")

            self.sequential_commands = []
            for i, cmd_config in enumerate(sequential_commands):
                if not isinstance(cmd_config, dict):
                    raise ValueError(f"Command {i} must be a dictionary.")

                # Required fields
                if "command" not in cmd_config:
                    raise ValueError(f"Command {i} missing required 'command' field.")

                command_base = cmd_config["command"]
                hydra_args = cmd_config.get("hydra_args", {})

                # Build hydra modifiers if provided
                hydra_modifiers = []
                for hydra_arg, value in hydra_args.items():
                    if hydra_arg.startswith("--"):
                        hydra_modifiers.append(f" {hydra_arg} {value}")
                    else:
                        hydra_modifiers.append(f" {hydra_arg}={_format_hydra_value(value)}")

                full_command = f"{command_base}{''.join(hydra_modifiers)}"

                self.sequential_commands.append(full_command)
                print(f"Prepared sequential command {i+1}: {full_command}")

            print(f"Built {len(self.sequential_commands)} sequential commands.")

        elif sequential_commands_list is not None:
            # Logic for job array of sequential commands
            if self.slurm_array_nb_jobs is not None:
                raise ValueError(
                    "Cannot provide slurm_array_nb_jobs during initialization when using sequential_commands_list."
                )
            if (
                not isinstance(sequential_commands_list, list)
                or not sequential_commands_list
            ):
                raise ValueError("sequential_commands_list must be a non-empty list.")

            self.sequential_commands_list = []
            for job_idx, sequential_commands in enumerate(sequential_commands_list):
                if not isinstance(sequential_commands, list) or not sequential_commands:
                    raise ValueError(
                        f"Job {job_idx} sequential_commands must be a non-empty list."
                    )

                job_sequential_commands = []
                for i, cmd_config in enumerate(sequential_commands):
                    if not isinstance(cmd_config, dict):
                        raise ValueError(
                            f"Job {job_idx}, Command {i} must be a dictionary."
                        )

                    # Required fields
                    if "command" not in cmd_config:
                        raise ValueError(
                            f"Job {job_idx}, Command {i} missing required 'command' field."
                        )

                    command_base = cmd_config["command"]
                    hydra_args = cmd_config.get("hydra_args", {})

                    # Build hydra modifiers if provided
                    hydra_modifiers = []
                    for hydra_arg, value in hydra_args.items():
                        if hydra_arg.startswith("--"):
                            hydra_modifiers.append(f" {hydra_arg} {value}")
                        else:
                            hydra_modifiers.append(f" {hydra_arg}={_format_hydra_value(value)}")

                    full_command = f"{command_base}{''.join(hydra_modifiers)}"

                    job_sequential_commands.append(full_command)

                self.sequential_commands_list.append(job_sequential_commands)

            # Set slurm_array_nb_jobs based on the list length
            self.slurm_array_nb_jobs = len(self.sequential_commands_list)
            print(
                f"Built {self.slurm_array_nb_jobs} job array entries with sequential commands."
            )

        elif hydra_args is not None:
            # Existing logic for single command or numeric array
            # If slurm_array_nb_jobs is provided in init, it defines a numeric array
            # If slurm_array_nb_jobs is None, it's a single command job
            if self.slurm_array_nb_jobs is not None and not isinstance(
                self.slurm_array_nb_jobs, int
            ):
                raise ValueError(
                    "If hydra_args is provided, slurm_array_nb_jobs (from init) must be an integer or None."
                )
            hydra_modifiers = []
            for hydra_arg, value in hydra_args.items():
                if hydra_arg.startswith("--"):
                    hydra_modifiers.append(f" {hydra_arg} {value}")
                else:
                    hydra_modifiers.append(f" {hydra_arg}={_format_hydra_value(value)}")
            self.cmd = f"{self.cmd_path} {''.join(hydra_modifiers)}"
            if self.slurm_array_nb_jobs is not None:
                print(
                    f"Built base command for numeric array ({self.slurm_array_nb_jobs} jobs): srun python {self.cmd}"
                )
            else:
                print(f"Built single command: srun python {self.cmd}")

    def launch(self, debug=False, dependency=None):
        if debug:
            self.time = "01:00:00"
            self.min_time = None
        # Check if commands have been built
        if (
            not hasattr(self, "cmd")
            and not hasattr(self, "cmds")
            and not hasattr(self, "sequential_commands")
            and not hasattr(self, "sequential_commands_list")
        ):
            raise ValueError("Run build_cmd first")
        if (
            self.cmd is None
            and self.cmds is None
            and self.sequential_commands is None
            and self.sequential_commands_list is None
        ):
            raise ValueError("Run build_cmd first - no command generated.")

        slurm_partition_directive = ""
        self.max_array_size = 10000  # Default max array size

        # When using torchrun, we only need 1 task per node (torchrun handles multi-GPU)
        ntasks_per_node = 1 if self.use_torchrun else self.num_gpus_per_node
        cpus_per_task = 224 // ntasks_per_node   # 224 CPUs / 8 GPUs per node
        # Build the python launcher command
        if self.use_torchrun:
            if self.nodes == 1:
                python_launcher = f"torchrun --standalone --nproc_per_node={self.num_gpus_per_node}"
            else:
                python_launcher = (
                    f"torchrun"
                    f" --nnodes={self.nodes}"
                    f" --nproc_per_node={self.num_gpus_per_node}"
                    f" --rdzv_backend=c10d"
                    f" --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
                )
        else:
            python_launcher = "python"

        slurm_partition_directive = f"#SBATCH --partition={self.partition}"
        print(f"Launching on partition {self.partition}")

        local_slurmfolder = Path("miro/checkpoints") / Path(self.expname) / Path("slurm")
        local_slurmfolder.mkdir(parents=True, exist_ok=True)

        array_string = ""
        job_suffix = ""

        if isinstance(self.slurm_array_nb_jobs, int):
            total_jobs = self.slurm_array_nb_jobs
            if total_jobs <= 0:
                array_string = ""
            elif total_jobs > self.max_array_size:
                num_chunks = math.ceil(total_jobs / self.max_array_size)
                if not (0 <= self.sub_job_index < num_chunks):
                    raise ValueError(
                        f"sub_job_index ({self.sub_job_index}) out of range for {num_chunks} chunks."
                    )

                start_index = self.sub_job_index * self.max_array_size
                end_index = (
                    min((self.sub_job_index + 1) * self.max_array_size, total_jobs) - 1
                )
                array_string = f"{start_index}-{end_index}"
                job_suffix = f"_chunk{self.sub_job_index}"
                print(
                    f"Launching chunk {self.sub_job_index}: jobs {start_index}-{end_index}"
                )
            else:
                if self.sub_job_index != 0:
                    print(
                        f"Warning: sub_job_index is {self.sub_job_index} but only one chunk is needed. Using index 0."
                    )
                    self.sub_job_index = 0
                array_string = f"0-{total_jobs - 1}"

        elif self.slurm_array_nb_jobs is not None:
            raise ValueError("slurm_array_nb_jobs must be an int or None.")

        if array_string and self.max_simultaneous_jobs is not None:
            array_string += f"%{self.max_simultaneous_jobs}"

        sbatch_array = f"#SBATCH --array={array_string}" if array_string else ""

        current_job_name = f"{self.job_name}{job_suffix}"
        slurm_path = local_slurmfolder / f"job_file{job_suffix}.slurm"

        # Store the script path
        self.slurm_script_path = slurm_path

        # Construct and store output/error paths with separate directories per task.
        # ``MIRO_SLURM_OUTPUT_DIR`` overrides where the logs land; the historical
        # default keeps the original layout for users who want it.
        slurm_output_base_dir = f"{DEFAULT_SLURM_OUTPUT_DIR}/{local_slurmfolder}/job_%j{job_suffix}"
        # Use %a for array task ID to create per-task subdirectories
        self.slurm_out_path = f"{slurm_output_base_dir}/task_%a/std.out"
        self.slurm_err_path = f"{slurm_output_base_dir}/task_%a/std.err"

        # Prepare commands for the SLURM script
        srun_command_line = ""
        bash_definitions = ""
        if self.sequential_commands:
            # Sequential commands
            command_lines = []
            for i, cmd in enumerate(self.sequential_commands):
                command_lines.append(
                    f"echo 'Executing command {i+1}/{len(self.sequential_commands)}: {cmd}'"
                )
                command_lines.append(f"srun uv run {python_launcher} {cmd}")
                command_lines.append("")

            srun_command_line = "\n".join(command_lines)

        elif self.sequential_commands_list:
            # Job array of sequential commands
            # Use SLURM_ARRAY_TASK_ID to select which set of sequential commands to run
            command_lines = []
            command_lines.append("# Select sequential commands based on array task ID")
            command_lines.append("TASK_ID=${SLURM_ARRAY_TASK_ID}")
            command_lines.append('echo "Running job array task $TASK_ID"')
            command_lines.append("")

            # Generate conditional blocks for each job in the array
            for job_idx, job_sequential_commands in enumerate(
                self.sequential_commands_list
            ):
                if job_idx == 0:
                    command_lines.append(f"if [ $TASK_ID -eq {job_idx} ]; then")
                else:
                    command_lines.append(f"elif [ $TASK_ID -eq {job_idx} ]; then")

                command_lines.append(
                    f"    echo 'Executing job {job_idx} with {len(job_sequential_commands)} sequential commands'"
                )

                for i, cmd in enumerate(job_sequential_commands):
                    command_lines.append(
                        f"    echo 'Command {i+1}/{len(job_sequential_commands)}: {cmd}'"
                    )
                    command_lines.append(f"    srun uv run {python_launcher} {cmd}")
                    if (
                        i < len(job_sequential_commands) - 1
                    ):  # Add blank line between commands except after last
                        command_lines.append("")

            command_lines.append("else")
            command_lines.append("    echo 'Error: Invalid SLURM_ARRAY_TASK_ID'")
            command_lines.append("    exit 1")
            command_lines.append("fi")

            srun_command_line = "\n".join(command_lines)

        elif self.cmds:
            # Create a bash array definition, quoting each command safely
            quoted_cmds = [shlex.quote(cmd) for cmd in self.cmds]
            bash_definitions = f"CMDS=({' '.join(quoted_cmds)})"
            # Use the SLURM_ARRAY_TASK_ID to index into the bash array
            # Adjust task ID based on the start index of the chunk
            start_index = 0
            if self.slurm_array_nb_jobs > self.max_array_size:
                # Ensure sub_job_index and max_array_size are defined and valid before using
                if self.sub_job_index is None or self.max_array_size is None:
                    raise ValueError(
                        "sub_job_index and max_array_size must be set for chunked arrays."
                    )
                start_index = self.sub_job_index * self.max_array_size

            srun_command_line = (
                f"COMMAND_INDEX=$((SLURM_ARRAY_TASK_ID - {start_index}))\n"
                f"echo 'Running command index $COMMAND_INDEX: ${{CMDS[$COMMAND_INDEX]}}'\n"
                f"srun uv run {python_launcher} ${{CMDS[$COMMAND_INDEX]}}"
            )

        elif self.cmd:
            # Single command execution
            srun_command_line = f"srun uv run {python_launcher} {self.cmd}"
        else:
            # This case should ideally not be reached if build_cmd was called
            raise ValueError(
                "No command or command list was built. Call build_cmd first."
            )

        slurm = f"""#!/bin/bash
#SBATCH --job-name={current_job_name}
{sbatch_array}
#SBATCH --nodes={self.nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --gres=gpu:{self.num_gpus_per_node}
{slurm_partition_directive}
#SBATCH --cpus-per-task={cpus_per_task}
{f"#SBATCH --time={self.time}" if self.time is not None else ""}
{f"#SBATCH --time-min={self.min_time}" if self.min_time is not None else ""}
#SBATCH --output={self.slurm_out_path}
#SBATCH --error={self.slurm_err_path}
#SBATCH --signal=SIGUSR1@180
{f"#SBATCH --export=NONE" if self.launch_from_compute_node else ""}

{f"unset SLURM_EXPORT_ENV" if self.launch_from_compute_node else ""}

export PYTHONPATH={self.root_dir if self.root_dir else DEFAULT_PYTHONPATH}:$PYTHONPATH

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((29500 + RANDOM % 1000))

export HYDRA_FULL_ERROR=1
# Keep torch recompilation logs opt-in; forcing them can hurt throughput.
if [ -n "${{TORCH_LOGS:-}}" ]; then
    export TORCH_LOGS
fi
export IS_CLUSTER=True

{f"unset SLURM_CPU_BIND" if self.launch_from_compute_node else ""}

# Define commands if using command list
{bash_definitions}

{f"# Stagger array job starts to avoid filesystem contention" if self.stagger_delay else ""}
{f"DELAY=$((RANDOM % {self.stagger_delay}))" if self.stagger_delay else ""}
{f'echo "Sleeping $DELAY seconds to stagger start..."' if self.stagger_delay else ""}
{f"sleep $DELAY" if self.stagger_delay else ""}

set -x
# Execute the appropriate command
{srun_command_line}
        """
        with open(slurm_path, "w") as slurm_file:
            slurm_file.write(slurm)
        # if self.launch_from_compute_node:
        #     os.system('unset $(env | egrep "SLURM_|SBATCH_"| cut -d= -f1)')
        print(f"Submitting SLURM script: {self.slurm_script_path}")
        dep_flag = f" --dependency={dependency}" if dependency else ""
        result = os.popen(f"sbatch{dep_flag} {self.slurm_script_path}").read()
        print(result)  # Print sbatch output (e.g., "Submitted batch job 12345")
        try:
            # Attempt to parse job ID
            self.slurm_job_id = int(result.strip().split()[-1])
            print(f"SLURM Job ID: {self.slurm_job_id}")
        except (IndexError, ValueError):
            print("Could not parse SLURM job ID from sbatch output.")


# export TRANSFORMERS_OFFLINE=1 # to avoid downloading
# export HYDRA_FULL_ERROR=1 # to have the full traceback
# export WANDB_CACHE_DIR=$NEWSCRATCH/wandb_cache
# export TMPDIR=$JOBSCRATCH
# export HF_HUB_OFFLINE=1
# export WANDB_MODE=offline
# export TORCH_LOGS=recompiles