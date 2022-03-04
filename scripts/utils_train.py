import os
import subprocess

def working_dir_setting(save_dir, name):
    try:
        os.mkdir(f'{save_dir}/{name}')
    except:
        i = 2
        while True:
            try:
                os.mkdir(f'{save_dir}/{name}{i}')
            except:
                i+=1
                continue
            else:
                save_dir=f'{save_dir}/{name}{i}'
                break
    else:
        save_dir = f'{save_dir}/{name}'
    return save_dir


class logger() :
    def __init__(self, log_file) :
        self.log_file = log_file
        try :
            with open(self.log_file, 'w') as w :
                pass
            print(f"\nStart logging")
            print(f"  log file path: {os.getcwd()}/{log_file}\n")
        except :
            print("Invalid argument 'save_dir'")
            exit(1)

    def __call__(self, *log, end='\n',save_log = True) :
        if len(log)==0:
            log = ('')
        log = '\n'.join([str(i) for i in log])
        print(log,end=end)
        if save_log :
            self.save(log+end)

    def log_arguments(self, args, put_log = True):
        args = vars(args)
        for a in args:
            if a == 'logger': continue
            print(f'{a}={args[a]}')
        if put_log :
            for a in args:
                if a =='logger': continue
                self.save(f'{a}={args[a]}\n')

    def save(self, log) :
        with open(self.log_file, 'a') as fw :
            fw.write(log)


def get_cuda_visible_devices(num_gpus: int) -> str:
        """Get available GPU IDs as a str (e.g., '0,1,2')"""
        max_num_gpus = 4
        idle_gpus = []

        if num_gpus:
            for i in range(max_num_gpus):
                cmd = ["nvidia-smi", "-i", str(i)]

                import sys
                major, minor = sys.version_info[0], sys.version_info[1]
                if major == 3 and minor > 6:
                    proc = subprocess.run(cmd, capture_output=True, text=True)  # after python 3.7
                if major == 3 and minor <= 6:
                    proc = subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True)  # for python 3.6

                if "No devices were found" in proc.stdout:
                    break

                if "No running" in proc.stdout:
                    idle_gpus.append(i)

                if len(idle_gpus) >= num_gpus:
                    break

            if len(idle_gpus) < num_gpus:
                msg = "Avaliable GPUs are less than required!"
                msg += f" ({num_gpus} required, {len(idle_gpus)} available)"
                raise RuntimeError(msg)

            # Convert to a str to feed to os.environ.
            idle_gpus = ",".join(str(i) for i in idle_gpus[:num_gpus])

        else:
            idle_gpus = ""

        return idle_gpus
