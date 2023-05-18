
import subprocess
import matplotlib.pyplot as plt

logs = ["log_adamw.txt", "log_default_lr_precondie-3_nolrdecay.txt"]
names = ["adamw", "psgd"]
legends = []
for i, log in enumerate(logs):
    results = subprocess.check_output(["grep", "val", log]).decode("UTF-8").rstrip().split("\n")
    step, train_loss, val_loss = [], [], []
    for line in results:
        one_line_list = line.split()
        step.append(float(one_line_list[1][:-1]))
        train_loss.append(float(one_line_list[4][:-1]))
        val_loss.append(float(one_line_list[7]))
    plt.plot(step, train_loss)
    plt.plot(step, val_loss)
    legends += [f"{names[i]} train", f"{names[i]} val"]

plt.title("nanoGPT loss")
plt.legend(legends)
plt.show()
